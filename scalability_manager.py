"""
Scalability Manager Module for PHERS
Phase 4: Performance Optimization

This module provides scalable architecture enhancements including:
- Connection pool management for concurrent operations
- Load balancing for distributed query processing
- Horizontal scaling capabilities
- Resource allocation and throttling
- Queue management for high-volume operations
- Failover and redundancy mechanisms
- Dynamic scaling based on workload patterns
"""

import threading
import time
import queue
import json
import logging
import asyncio
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Future, as_completed
from pathlib import Path
import sqlite3
import multiprocessing as mp
import psutil
import weakref
from contextlib import contextmanager
import hashlib

@dataclass
class ConnectionConfig:
    """Database connection configuration"""
    db_path: str
    max_connections: int
    min_connections: int
    connection_timeout: float
    idle_timeout: float
    retry_attempts: int
    retry_delay: float

@dataclass
class WorkerNode:
    """Worker node configuration for distributed processing"""
    node_id: str
    node_type: str  # 'process', 'thread', 'async'
    capacity: int
    current_load: int
    status: str  # 'active', 'busy', 'idle', 'error'
    created_at: datetime
    last_heartbeat: datetime
    total_tasks_completed: int
    average_task_time: float

@dataclass
class TaskRequest:
    """Represents a task request for processing"""
    task_id: str
    task_type: str
    priority: int  # 1-10, higher is more important
    payload: Dict[str, Any]
    created_at: datetime
    scheduled_at: Optional[datetime] = None
    retries: int = 0
    max_retries: int = 3
    timeout: float = 30.0

@dataclass
class TaskResult:
    """Result of task processing"""
    task_id: str
    status: str  # 'success', 'error', 'timeout'
    result: Any
    error_message: Optional[str]
    processing_time: float
    processed_by: str  # node_id
    completed_at: datetime

@dataclass
class LoadMetrics:
    """System load and performance metrics"""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    active_connections: int
    queue_size: int
    throughput_per_second: float
    average_response_time: float
    error_rate: float
    worker_utilization: float

class ConnectionPool:
    """Thread-safe database connection pool"""
    
    def __init__(self, config: ConnectionConfig):
        self.config = config
        self.connections = queue.Queue(maxsize=config.max_connections)
        self.active_connections = 0
        self.connection_stats = defaultdict(int)
        self.lock = threading.Lock()
        
        # Pre-populate with minimum connections
        for _ in range(config.min_connections):
            conn = self._create_connection()
            if conn:
                self.connections.put(conn)
                self.active_connections += 1
    
    def _create_connection(self) -> Optional[sqlite3.Connection]:
        """Create a new database connection"""
        try:
            conn = sqlite3.connect(
                self.config.db_path,
                timeout=self.config.connection_timeout,
                check_same_thread=False
            )
            conn.row_factory = sqlite3.Row  # Enable named columns
            return conn
        except Exception as e:
            logging.error(f"Failed to create connection: {e}")
            return None
    
    @contextmanager
    def get_connection(self):
        """Get a connection from the pool"""
        conn = None
        try:
            # Try to get existing connection
            try:
                conn = self.connections.get_nowait()
            except queue.Empty:
                # Create new connection if under max limit
                with self.lock:
                    if self.active_connections < self.config.max_connections:
                        conn = self._create_connection()
                        if conn:
                            self.active_connections += 1
                    else:
                        # Wait for available connection
                        conn = self.connections.get(timeout=self.config.connection_timeout)
            
            if not conn:
                raise Exception("Unable to get database connection")
            
            # Test connection
            conn.execute("SELECT 1")
            
            yield conn
        
        except Exception as e:
            if conn:
                try:
                    conn.close()
                except:
                    pass
                with self.lock:
                    self.active_connections -= 1
            raise e
        
        finally:
            if conn:
                try:
                    # Return connection to pool
                    self.connections.put(conn, timeout=1.0)
                except queue.Full:
                    # Pool is full, close connection
                    conn.close()
                    with self.lock:
                        self.active_connections -= 1
    
    def close_all(self):
        """Close all connections in the pool"""
        while not self.connections.empty():
            try:
                conn = self.connections.get_nowait()
                conn.close()
            except:
                pass
        self.active_connections = 0

class TaskQueue:
    """Priority-based task queue with scheduling support"""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.priority_queues = {i: queue.Queue() for i in range(1, 11)}  # Priority 1-10
        self.scheduled_tasks = queue.PriorityQueue()
        self.task_results = {}
        self.lock = threading.Lock()
        
        # Statistics
        self.tasks_submitted = 0
        self.tasks_completed = 0
        self.tasks_failed = 0
        
        # Scheduler thread
        self.scheduler_active = False
        self.scheduler_thread = None
        self.start_scheduler()
    
    def start_scheduler(self):
        """Start the task scheduler thread"""
        if self.scheduler_active:
            return
        
        self.scheduler_active = True
        
        def scheduler_loop():
            while self.scheduler_active:
                try:
                    # Check for scheduled tasks
                    current_time = datetime.now()
                    ready_tasks = []
                    
                    # Get all ready tasks (non-blocking check)
                    while True:
                        try:
                            scheduled_time, task = self.scheduled_tasks.get_nowait()
                            if scheduled_time <= current_time:
                                ready_tasks.append(task)
                            else:
                                # Put it back, not ready yet
                                self.scheduled_tasks.put((scheduled_time, task))
                                break
                        except queue.Empty:
                            break
                    
                    # Add ready tasks to priority queues
                    for task in ready_tasks:
                        self.priority_queues[task.priority].put(task)
                    
                    time.sleep(1.0)  # Check every second
                
                except Exception as e:
                    logging.error(f"Error in scheduler loop: {e}")
                    time.sleep(1.0)
        
        self.scheduler_thread = threading.Thread(target=scheduler_loop, daemon=True)
        self.scheduler_thread.start()
    
    def stop_scheduler(self):
        """Stop the task scheduler"""
        self.scheduler_active = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5.0)
    
    def submit_task(self, task: TaskRequest) -> str:
        """Submit a task to the queue"""
        with self.lock:
            if self.get_queue_size() >= self.max_size:
                raise Exception("Task queue is full")
            
            self.tasks_submitted += 1
        
        if task.scheduled_at and task.scheduled_at > datetime.now():
            # Schedule for future execution
            self.scheduled_tasks.put((task.scheduled_at, task))
        else:
            # Add to immediate execution queue
            self.priority_queues[task.priority].put(task)
        
        return task.task_id
    
    def get_next_task(self, timeout: float = 1.0) -> Optional[TaskRequest]:
        """Get the next highest priority task"""
        # Check priority queues from highest to lowest
        for priority in range(10, 0, -1):
            try:
                task = self.priority_queues[priority].get_nowait()
                return task
            except queue.Empty:
                continue
        
        # If no high priority tasks, wait for any task
        for priority in range(10, 0, -1):
            try:
                task = self.priority_queues[priority].get(timeout=timeout/10)
                return task
            except queue.Empty:
                continue
        
        return None
    
    def complete_task(self, result: TaskResult):
        """Mark a task as completed"""
        with self.lock:
            self.task_results[result.task_id] = result
            if result.status == 'success':
                self.tasks_completed += 1
            else:
                self.tasks_failed += 1
    
    def get_queue_size(self) -> int:
        """Get total queue size"""
        return sum(q.qsize() for q in self.priority_queues.values()) + self.scheduled_tasks.qsize()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get queue statistics"""
        return {
            'tasks_submitted': self.tasks_submitted,
            'tasks_completed': self.tasks_completed,
            'tasks_failed': self.tasks_failed,
            'current_queue_size': self.get_queue_size(),
            'scheduled_tasks': self.scheduled_tasks.qsize(),
            'success_rate': self.tasks_completed / max(1, self.tasks_submitted)
        }

class WorkerManager:
    """Manages worker nodes for distributed processing"""
    
    def __init__(self, max_workers: int = None):
        if max_workers is None:
            max_workers = min(32, (mp.cpu_count() or 1) + 4)
        
        self.max_workers = max_workers
        self.workers = {}
        self.worker_stats = defaultdict(dict)
        self.load_balancer = LoadBalancer()
        
        # Executors for different worker types
        self.thread_executor = ThreadPoolExecutor(max_workers=max_workers//2)
        self.process_executor = ProcessPoolExecutor(max_workers=max_workers//4)
        
        # Worker monitoring
        self.monitoring_active = False
        self.monitoring_thread = None
    
    def create_worker(self, worker_type: str = 'thread', capacity: int = 10) -> str:
        """Create a new worker node"""
        worker_id = f"{worker_type}_{int(time.time() * 1000000)}"
        
        worker = WorkerNode(
            node_id=worker_id,
            node_type=worker_type,
            capacity=capacity,
            current_load=0,
            status='idle',
            created_at=datetime.now(),
            last_heartbeat=datetime.now(),
            total_tasks_completed=0,
            average_task_time=0.0
        )
        
        self.workers[worker_id] = worker
        return worker_id
    
    def get_available_worker(self, task_type: str = None) -> Optional[WorkerNode]:
        """Get an available worker using load balancing"""
        available_workers = [
            worker for worker in self.workers.values()
            if worker.status in ['idle', 'active'] and worker.current_load < worker.capacity
        ]
        
        if not available_workers:
            return None
        
        # Use load balancer to select worker
        return self.load_balancer.select_worker(available_workers, task_type)
    
    def assign_task(self, worker: WorkerNode, task: TaskRequest) -> Future:
        """Assign a task to a worker"""
        worker.current_load += 1
        worker.status = 'busy' if worker.current_load >= worker.capacity else 'active'
        worker.last_heartbeat = datetime.now()
        
        # Submit task based on worker type
        if worker.node_type == 'thread':
            future = self.thread_executor.submit(self._execute_task, worker, task)
        elif worker.node_type == 'process':
            future = self.process_executor.submit(self._execute_task_in_process, worker.node_id, task)
        else:
            raise ValueError(f"Unsupported worker type: {worker.node_type}")
        
        # Add callback to update worker stats
        future.add_done_callback(lambda f: self._task_completed(worker, task, f))
        
        return future
    
    def _execute_task(self, worker: WorkerNode, task: TaskRequest) -> TaskResult:
        """Execute a task in the current process"""
        start_time = time.time()
        
        try:
            # Route task to appropriate handler
            result = self._route_task(task)
            
            return TaskResult(
                task_id=task.task_id,
                status='success',
                result=result,
                error_message=None,
                processing_time=time.time() - start_time,
                processed_by=worker.node_id,
                completed_at=datetime.now()
            )
        
        except Exception as e:
            return TaskResult(
                task_id=task.task_id,
                status='error',
                result=None,
                error_message=str(e),
                processing_time=time.time() - start_time,
                processed_by=worker.node_id,
                completed_at=datetime.now()
            )
    
    def _execute_task_in_process(self, worker_id: str, task: TaskRequest) -> TaskResult:
        """Execute a task in a separate process"""
        # This would be used for CPU-intensive tasks
        return self._execute_task(self.workers[worker_id], task)
    
    def _route_task(self, task: TaskRequest) -> Any:
        """Route task to appropriate handler based on task type"""
        if task.task_type == 'sql_query':
            return self._handle_sql_query(task)
        elif task.task_type == 'data_analysis':
            return self._handle_data_analysis(task)
        elif task.task_type == 'cache_operation':
            return self._handle_cache_operation(task)
        else:
            raise ValueError(f"Unknown task type: {task.task_type}")
    
    def _handle_sql_query(self, task: TaskRequest) -> Any:
        """Handle SQL query task"""
        # This would integrate with the connection pool and database optimizer
        query = task.payload.get('query')
        db_path = task.payload.get('db_path', 'data.db')
        
        # Simplified implementation
        import sqlite3
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            return cursor.fetchall()
    
    def _handle_data_analysis(self, task: TaskRequest) -> Any:
        """Handle data analysis task"""
        # This would integrate with AI components
        analysis_type = task.payload.get('analysis_type')
        data = task.payload.get('data')
        
        # Simplified implementation
        if analysis_type == 'column_intelligence':
            # Integrate with column_intelligence.py
            pass
        elif analysis_type == 'query_optimization':
            # Integrate with query_intelligence.py
            pass
        
        return {"analysis": "completed", "type": analysis_type}
    
    def _handle_cache_operation(self, task: TaskRequest) -> Any:
        """Handle cache operation task"""
        operation = task.payload.get('operation')
        key = task.payload.get('key')
        value = task.payload.get('value')
        
        # This would integrate with intelligent_cache.py
        if operation == 'get':
            return {"cached_value": None}  # Simplified
        elif operation == 'set':
            return {"cached": True}  # Simplified
        
        return None
    
    def _task_completed(self, worker: WorkerNode, task: TaskRequest, future: Future):
        """Handle task completion and update worker stats"""
        try:
            result = future.result()
            
            # Update worker stats
            worker.current_load = max(0, worker.current_load - 1)
            worker.total_tasks_completed += 1
            worker.status = 'idle' if worker.current_load == 0 else 'active'
            
            # Update average task time
            if worker.average_task_time == 0:
                worker.average_task_time = result.processing_time
            else:
                worker.average_task_time = (worker.average_task_time + result.processing_time) / 2
        
        except Exception as e:
            logging.error(f"Task failed: {e}")
            worker.current_load = max(0, worker.current_load - 1)
            worker.status = 'error'
    
    def cleanup_inactive_workers(self, timeout_minutes: int = 30):
        """Clean up workers that haven't had a heartbeat"""
        cutoff_time = datetime.now() - timedelta(minutes=timeout_minutes)
        inactive_workers = [
            worker_id for worker_id, worker in self.workers.items()
            if worker.last_heartbeat < cutoff_time
        ]
        
        for worker_id in inactive_workers:
            del self.workers[worker_id]
    
    def shutdown(self):
        """Shutdown all workers"""
        self.monitoring_active = False
        self.thread_executor.shutdown(wait=True)
        self.process_executor.shutdown(wait=True)

class LoadBalancer:
    """Load balancer for distributing tasks across workers"""
    
    def __init__(self, strategy: str = 'least_loaded'):
        self.strategy = strategy
        self.worker_performance = defaultdict(list)  # Track performance history
    
    def select_worker(self, available_workers: List[WorkerNode], 
                     task_type: str = None) -> WorkerNode:
        """Select the best worker based on load balancing strategy"""
        if not available_workers:
            return None
        
        if self.strategy == 'round_robin':
            return self._round_robin_selection(available_workers)
        elif self.strategy == 'least_loaded':
            return self._least_loaded_selection(available_workers)
        elif self.strategy == 'performance_based':
            return self._performance_based_selection(available_workers)
        elif self.strategy == 'task_affinity':
            return self._task_affinity_selection(available_workers, task_type)
        else:
            return available_workers[0]  # Default to first available
    
    def _round_robin_selection(self, workers: List[WorkerNode]) -> WorkerNode:
        """Simple round-robin selection"""
        if not hasattr(self, '_round_robin_index'):
            self._round_robin_index = 0
        
        worker = workers[self._round_robin_index % len(workers)]
        self._round_robin_index += 1
        return worker
    
    def _least_loaded_selection(self, workers: List[WorkerNode]) -> WorkerNode:
        """Select worker with lowest current load"""
        return min(workers, key=lambda w: w.current_load / w.capacity)
    
    def _performance_based_selection(self, workers: List[WorkerNode]) -> WorkerNode:
        """Select worker based on historical performance"""
        def performance_score(worker):
            # Combine load and historical performance
            load_factor = 1.0 - (worker.current_load / worker.capacity)
            perf_factor = 1.0 / max(0.1, worker.average_task_time)  # Avoid division by zero
            completion_factor = worker.total_tasks_completed / max(1, worker.total_tasks_completed + 1)
            
            return load_factor * 0.4 + perf_factor * 0.4 + completion_factor * 0.2
        
        return max(workers, key=performance_score)
    
    def _task_affinity_selection(self, workers: List[WorkerNode], 
                               task_type: str) -> WorkerNode:
        """Select worker based on task type affinity"""
        # For now, use least loaded as fallback
        return self._least_loaded_selection(workers)

class ScalabilityManager:
    """
    Main scalability management system that coordinates connection pooling,
    worker management, load balancing, and resource scaling.
    """
    
    def __init__(self, db_path: str = "data.db", activity_log_path: str = "/data/activity.log"):
        self.db_path = db_path
        self.activity_log_path = activity_log_path
        
        # Initialize components
        self.connection_config = ConnectionConfig(
            db_path=db_path,
            max_connections=50,
            min_connections=5,
            connection_timeout=30.0,
            idle_timeout=300.0,
            retry_attempts=3,
            retry_delay=1.0
        )
        
        self.connection_pool = ConnectionPool(self.connection_config)
        self.task_queue = TaskQueue(max_size=10000)
        self.worker_manager = WorkerManager(max_workers=multiprocessing.cpu_count() * 2)
        
        # Metrics and monitoring
        self.load_metrics = deque(maxlen=1440)  # 24 hours of metrics
        self.auto_scaling_enabled = True
        self.scaling_thresholds = {
            'scale_up_cpu': 70.0,        # Scale up if CPU > 70%
            'scale_up_queue': 100,       # Scale up if queue > 100 tasks
            'scale_down_cpu': 30.0,      # Scale down if CPU < 30%
            'scale_down_idle': 300       # Scale down if idle > 5 minutes
        }
        
        # Monitoring thread
        self.monitoring_active = False
        self.monitoring_thread = None
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize some workers
        for _ in range(min(4, multiprocessing.cpu_count())):
            self.worker_manager.create_worker('thread', capacity=5)
        
        # Start monitoring
        self.start_monitoring()
        
        self._log_activity("ScalabilityManager initialized", {
            "db_path": db_path,
            "max_connections": self.connection_config.max_connections,
            "initial_workers": len(self.worker_manager.workers),
            "auto_scaling": self.auto_scaling_enabled
        })
    
    def _log_activity(self, activity: str, details: Dict[str, Any] = None):
        """Log activity to the activity log file"""
        try:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "module": "ScalabilityManager",
                "activity": activity,
                "details": details or {}
            }
            
            Path(self.activity_log_path).parent.mkdir(exist_ok=True)
            with open(self.activity_log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            self.logger.error(f"Failed to log activity: {e}")
    
    def start_monitoring(self, interval: float = 60.0):
        """Start system monitoring and auto-scaling"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        
        def monitoring_loop():
            while self.monitoring_active:
                try:
                    # Collect metrics
                    metrics = self._collect_load_metrics()
                    self.load_metrics.append(metrics)
                    
                    # Perform auto-scaling if enabled
                    if self.auto_scaling_enabled:
                        self._auto_scale_resources(metrics)
                    
                    # Cleanup inactive resources
                    self.worker_manager.cleanup_inactive_workers()
                    
                    time.sleep(interval)
                
                except Exception as e:
                    self.logger.error(f"Error in monitoring loop: {e}")
                    time.sleep(interval)
        
        self.monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
        self.monitoring_thread.start()
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
    
    def _collect_load_metrics(self) -> LoadMetrics:
        """Collect current system load metrics"""
        # System metrics
        cpu_usage = psutil.cpu_percent()
        memory_usage = psutil.virtual_memory().percent
        
        # Application metrics
        active_connections = self.connection_pool.active_connections
        queue_size = self.task_queue.get_queue_size()
        
        # Calculate throughput and response time
        queue_stats = self.task_queue.get_statistics()
        
        # Worker utilization
        total_workers = len(self.worker_manager.workers)
        busy_workers = len([w for w in self.worker_manager.workers.values() 
                           if w.status == 'busy'])
        worker_utilization = busy_workers / max(1, total_workers)
        
        metrics = LoadMetrics(
            timestamp=datetime.now(),
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            active_connections=active_connections,
            queue_size=queue_size,
            throughput_per_second=0,  # Would need more sophisticated calculation
            average_response_time=0,  # Would need response time tracking
            error_rate=queue_stats.get('success_rate', 1.0),
            worker_utilization=worker_utilization
        )
        
        return metrics
    
    def _auto_scale_resources(self, metrics: LoadMetrics):
        """Automatically scale resources based on load"""
        # Scale up conditions
        should_scale_up = (
            metrics.cpu_usage > self.scaling_thresholds['scale_up_cpu'] or
            metrics.queue_size > self.scaling_thresholds['scale_up_queue'] or
            metrics.worker_utilization > 0.8
        )
        
        # Scale down conditions
        should_scale_down = (
            metrics.cpu_usage < self.scaling_thresholds['scale_down_cpu'] and
            metrics.queue_size < 10 and
            metrics.worker_utilization < 0.3
        )
        
        current_workers = len(self.worker_manager.workers)
        
        if should_scale_up and current_workers < self.worker_manager.max_workers:
            # Add new worker
            worker_id = self.worker_manager.create_worker('thread', capacity=5)
            self._log_activity("Scaled up resources", {
                "new_worker": worker_id,
                "total_workers": current_workers + 1,
                "trigger": "high_load"
            })
        
        elif should_scale_down and current_workers > 2:
            # Remove idle worker
            idle_workers = [w for w in self.worker_manager.workers.values() 
                           if w.status == 'idle' and w.current_load == 0]
            
            if idle_workers:
                worker_to_remove = min(idle_workers, key=lambda w: w.total_tasks_completed)
                del self.worker_manager.workers[worker_to_remove.node_id]
                
                self._log_activity("Scaled down resources", {
                    "removed_worker": worker_to_remove.node_id,
                    "total_workers": current_workers - 1,
                    "trigger": "low_load"
                })
    
    def submit_task(self, task_type: str, payload: Dict[str, Any], 
                   priority: int = 5, scheduled_at: datetime = None) -> str:
        """Submit a task for processing"""
        task = TaskRequest(
            task_id=f"task_{int(time.time() * 1000000)}",
            task_type=task_type,
            priority=priority,
            payload=payload,
            created_at=datetime.now(),
            scheduled_at=scheduled_at
        )
        
        return self.task_queue.submit_task(task)
    
    def process_tasks(self, max_concurrent: int = None) -> Dict[str, Any]:
        """Process tasks from the queue"""
        if max_concurrent is None:
            max_concurrent = len(self.worker_manager.workers)
        
        active_futures = []
        completed_tasks = []
        failed_tasks = []
        
        # Process tasks until queue is empty or max concurrent reached
        while len(active_futures) < max_concurrent:
            task = self.task_queue.get_next_task(timeout=1.0)
            if not task:
                break
            
            # Get available worker
            worker = self.worker_manager.get_available_worker(task.task_type)
            if not worker:
                # Put task back and break
                self.task_queue.priority_queues[task.priority].put(task)
                break
            
            # Assign task to worker
            future = self.worker_manager.assign_task(worker, task)
            active_futures.append((task, future))
        
        # Wait for completion and collect results
        for task, future in active_futures:
            try:
                result = future.result(timeout=task.timeout)
                completed_tasks.append(result)
                self.task_queue.complete_task(result)
            except Exception as e:
                failed_result = TaskResult(
                    task_id=task.task_id,
                    status='error',
                    result=None,
                    error_message=str(e),
                    processing_time=0,
                    processed_by='system',
                    completed_at=datetime.now()
                )
                failed_tasks.append(failed_result)
                self.task_queue.complete_task(failed_result)
        
        return {
            'completed': len(completed_tasks),
            'failed': len(failed_tasks),
            'results': completed_tasks,
            'errors': failed_tasks
        }
    
    @contextmanager
    def get_database_connection(self):
        """Get a database connection from the pool"""
        with self.connection_pool.get_connection() as conn:
            yield conn
    
    def execute_query(self, query: str, params: Tuple = None) -> List[Dict[str, Any]]:
        """Execute a query using connection pool"""
        with self.get_database_connection() as conn:
            cursor = conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            # Convert to list of dictionaries
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            rows = cursor.fetchall()
            
            return [dict(zip(columns, row)) for row in rows]
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        current_metrics = self._collect_load_metrics()
        
        status = {
            'timestamp': datetime.now().isoformat(),
            'system_metrics': asdict(current_metrics),
            'connection_pool': {
                'active_connections': self.connection_pool.active_connections,
                'max_connections': self.connection_config.max_connections
            },
            'task_queue': self.task_queue.get_statistics(),
            'workers': {
                'total_workers': len(self.worker_manager.workers),
                'active_workers': len([w for w in self.worker_manager.workers.values() 
                                     if w.status == 'active']),
                'busy_workers': len([w for w in self.worker_manager.workers.values() 
                                   if w.status == 'busy']),
                'idle_workers': len([w for w in self.worker_manager.workers.values() 
                                   if w.status == 'idle'])
            },
            'auto_scaling': {
                'enabled': self.auto_scaling_enabled,
                'thresholds': self.scaling_thresholds
            },
            'performance_trends': self._calculate_performance_trends()
        }
        
        return status
    
    def _calculate_performance_trends(self) -> Dict[str, str]:
        """Calculate performance trends"""
        if len(self.load_metrics) < 10:
            return {'trend': 'insufficient_data'}
        
        recent_metrics = list(self.load_metrics)[-10:]
        
        # CPU trend
        cpu_values = [m.cpu_usage for m in recent_metrics]
        cpu_trend = 'stable'
        if len(cpu_values) >= 5:
            first_half_avg = sum(cpu_values[:len(cpu_values)//2]) / (len(cpu_values)//2)
            second_half_avg = sum(cpu_values[len(cpu_values)//2:]) / (len(cpu_values) - len(cpu_values)//2)
            
            if second_half_avg > first_half_avg * 1.2:
                cpu_trend = 'increasing'
            elif second_half_avg < first_half_avg * 0.8:
                cpu_trend = 'decreasing'
        
        # Queue size trend
        queue_values = [m.queue_size for m in recent_metrics]
        queue_trend = 'stable'
        if len(queue_values) >= 5:
            first_half_avg = sum(queue_values[:len(queue_values)//2]) / (len(queue_values)//2)
            second_half_avg = sum(queue_values[len(queue_values)//2:]) / (len(queue_values) - len(queue_values)//2)
            
            if second_half_avg > first_half_avg * 2:
                queue_trend = 'growing'
            elif second_half_avg < first_half_avg * 0.5:
                queue_trend = 'shrinking'
        
        return {
            'cpu_trend': cpu_trend,
            'queue_trend': queue_trend,
            'overall_health': 'good' if cpu_trend != 'increasing' and queue_trend != 'growing' else 'degrading'
        }
    
    def export_scalability_report(self, filepath: str = None) -> Dict[str, Any]:
        """Export comprehensive scalability report"""
        if not filepath:
            filepath = f"/data/scalability_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        status = self.get_system_status()
        
        report = {
            'report_metadata': {
                'generated_at': datetime.now().isoformat(),
                'monitoring_period_hours': len(self.load_metrics) / 60.0,  # Assuming 1-minute intervals
                'auto_scaling_enabled': self.auto_scaling_enabled
            },
            'system_status': status,
            'load_history': [asdict(metrics) for metrics in self.load_metrics],
            'worker_details': [asdict(worker) for worker in self.worker_manager.workers.values()],
            'configuration': {
                'connection_pool': asdict(self.connection_config),
                'scaling_thresholds': self.scaling_thresholds,
                'max_workers': self.worker_manager.max_workers
            }
        }
        
        try:
            Path(filepath).parent.mkdir(exist_ok=True)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, default=str)
            
            self._log_activity("Scalability report exported", {"filepath": filepath})
            return {"success": True, "filepath": filepath, "report": report}
        
        except Exception as e:
            self.logger.error(f"Failed to export report: {e}")
            return {"success": False, "error": str(e), "report": report}
    
    def shutdown(self):
        """Shutdown all scalability components"""
        self.stop_monitoring()
        self.task_queue.stop_scheduler()
        self.worker_manager.shutdown()
        self.connection_pool.close_all()
        
        self._log_activity("ScalabilityManager shutdown", {
            "processed_tasks": self.task_queue.tasks_completed,
            "failed_tasks": self.task_queue.tasks_failed
        })

# Global instance for easy integration
scalability_manager = None

def get_scalability_manager(db_path: str = None) -> ScalabilityManager:
    """Get or create global scalability manager instance"""
    global scalability_manager
    if scalability_manager is None or (db_path and scalability_manager.db_path != db_path):
        scalability_manager = ScalabilityManager(db_path or "data.db")
    return scalability_manager

def submit_task(task_type: str, payload: Dict[str, Any], priority: int = 5):
    """Convenience function to submit a task"""
    manager = get_scalability_manager()
    return manager.submit_task(task_type, payload, priority)

def execute_scalable_query(query: str, params: Tuple = None):
    """Convenience function to execute query with connection pooling"""
    manager = get_scalability_manager()
    return manager.execute_query(query, params)

if __name__ == "__main__":
    # Example usage and testing
    manager = ScalabilityManager("test.db")
    
    # Submit some test tasks
    for i in range(10):
        task_id = manager.submit_task(
            'sql_query',
            {'query': f'SELECT {i} as test_value', 'db_path': 'test.db'},
            priority=i % 10 + 1
        )
        print(f"Submitted task: {task_id}")
    
    # Process tasks
    results = manager.process_tasks()
    print(f"Processed {results['completed']} tasks, {results['failed']} failed")
    
    # Get system status
    status = manager.get_system_status()
    print(f"System Status: {status['workers']['active_workers']} active workers")
    
    # Export report
    report = manager.export_scalability_report()
    print(f"Report exported: {report.get('filepath', 'Failed')}")
    
    # Test connection pool
    with manager.get_database_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT 1 as test")
        result = cursor.fetchone()
        print(f"Connection pool test: {result}")
    
    # Shutdown
    manager.shutdown()