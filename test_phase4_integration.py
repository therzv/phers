"""
Phase 4 Integration Tests for PHERS
Performance Optimization Module Testing

This module provides comprehensive testing for Phase 4 components:
- Intelligent caching system testing
- Database optimization testing
- Performance monitoring validation
- Memory management testing
- Scalability system testing
- Integration testing with previous phases
"""

import unittest
import time
import threading
import tempfile
import json
import sqlite3
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from contextlib import contextmanager
import multiprocessing

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from intelligent_cache import IntelligentCache, get_intelligent_cache
    from database_optimizer import DatabaseOptimizer, get_database_optimizer
    from performance_monitor import PerformanceMonitor, get_performance_monitor
    from memory_manager import MemoryManager, get_memory_manager
    from scalability_manager import ScalabilityManager, get_scalability_manager
    
    # Phase integration imports
    from column_intelligence import ColumnIntelligence
    from query_intelligence import QueryIntelligence
    from ai_query_generator import AIQueryGenerator
    
    PHASE4_IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some Phase 4 modules not available: {e}")
    PHASE4_IMPORTS_AVAILABLE = False

class TestIntelligentCache(unittest.TestCase):
    """Test intelligent caching system"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.test_db.close()
        
        # Create test database
        with sqlite3.connect(self.test_db.name) as conn:
            conn.execute('''
                CREATE TABLE test_data (
                    id INTEGER PRIMARY KEY,
                    name TEXT,
                    value INTEGER,
                    created_at TEXT
                )
            ''')
            
            # Insert test data
            for i in range(100):
                conn.execute(
                    "INSERT INTO test_data (name, value, created_at) VALUES (?, ?, ?)",
                    (f"item_{i}", i * 10, datetime.now().isoformat())
                )
        
        if PHASE4_IMPORTS_AVAILABLE:
            self.cache = IntelligentCache()
    
    def tearDown(self):
        """Clean up test environment"""
        try:
            os.unlink(self.test_db.name)
        except:
            pass
    
    @unittest.skipUnless(PHASE4_IMPORTS_AVAILABLE, "Phase 4 modules not available")
    def test_cache_basic_operations(self):
        """Test basic cache operations"""
        # Test set and get
        test_key = "test_query_123"
        test_value = {"result": [1, 2, 3], "metadata": {"count": 3}}
        
        success = self.cache.set(test_key, test_value, category="query_results")
        self.assertTrue(success)
        
        cached_value = self.cache.get(test_key, category="query_results")
        self.assertIsNotNone(cached_value)
        self.assertEqual(cached_value["result"], [1, 2, 3])
        self.assertEqual(cached_value["metadata"]["count"], 3)
    
    @unittest.skipUnless(PHASE4_IMPORTS_AVAILABLE, "Phase 4 modules not available")
    def test_cache_expiration(self):
        """Test cache expiration mechanism"""
        # Set item with short TTL
        test_key = "expiring_key"
        test_value = {"data": "expires_soon"}
        
        success = self.cache.set(test_key, test_value, category="query_results", ttl_override=1)
        self.assertTrue(success)
        
        # Should be available immediately
        cached_value = self.cache.get(test_key, category="query_results")
        self.assertIsNotNone(cached_value)
        
        # Wait for expiration
        time.sleep(2)
        
        # Should be None after expiration
        cached_value = self.cache.get(test_key, category="query_results")
        self.assertIsNone(cached_value)
    
    @unittest.skipUnless(PHASE4_IMPORTS_AVAILABLE, "Phase 4 modules not available")
    def test_cache_semantic_similarity(self):
        """Test semantic similarity matching"""
        # Cache a query result
        original_query = "SELECT name FROM test_data WHERE value > 50"
        result_data = {"results": ["item_6", "item_7"], "count": 2}
        
        query_key = self.cache._generate_semantic_cache_key(
            original_query, {"table": "test_data"}
        )
        self.cache.set(query_key, result_data, category="query_results")
        
        # Test similar query
        similar_query = "SELECT name FROM test_data WHERE value >= 60"
        similar_key = self.cache._generate_semantic_cache_key(
            similar_query, {"table": "test_data"}
        )
        
        # Should find similar cached result
        similar_results = self.cache.get_similar_cached_results(
            similar_query, {"table": "test_data"}, similarity_threshold=0.7
        )
        
        self.assertGreater(len(similar_results), 0)
    
    @unittest.skipUnless(PHASE4_IMPORTS_AVAILABLE, "Phase 4 modules not available")
    def test_cache_performance_metrics(self):
        """Test cache performance tracking"""
        # Perform several cache operations
        for i in range(10):
            key = f"perf_test_{i}"
            value = {"data": f"value_{i}"}
            self.cache.set(key, value, category="query_results")
        
        # Get some values (hits)
        for i in range(5):
            key = f"perf_test_{i}"
            self.cache.get(key, category="query_results")
        
        # Try to get non-existent values (misses)
        for i in range(5):
            key = f"missing_key_{i}"
            self.cache.get(key, category="query_results")
        
        # Check performance metrics
        metrics = self.cache.get_performance_metrics()
        self.assertGreater(metrics["total_hits"], 0)
        self.assertGreater(metrics["total_misses"], 0)
        self.assertGreater(metrics["hit_rate"], 0)

class TestDatabaseOptimizer(unittest.TestCase):
    """Test database optimization system"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.test_db.close()
        
        # Create test database with tables that need optimization
        with sqlite3.connect(self.test_db.name) as conn:
            # Users table
            conn.execute('''
                CREATE TABLE users (
                    id INTEGER PRIMARY KEY,
                    email TEXT UNIQUE,
                    name TEXT,
                    created_at TEXT
                )
            ''')
            
            # Orders table
            conn.execute('''
                CREATE TABLE orders (
                    id INTEGER PRIMARY KEY,
                    user_id INTEGER,
                    product_name TEXT,
                    amount REAL,
                    order_date TEXT,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
            ''')
            
            # Insert test data
            for i in range(1000):
                conn.execute(
                    "INSERT INTO users (email, name, created_at) VALUES (?, ?, ?)",
                    (f"user_{i}@test.com", f"User {i}", datetime.now().isoformat())
                )
            
            for i in range(5000):
                conn.execute(
                    "INSERT INTO orders (user_id, product_name, amount, order_date) VALUES (?, ?, ?, ?)",
                    (i % 1000 + 1, f"Product {i % 100}", (i % 100) * 10.99, datetime.now().isoformat())
                )
        
        if PHASE4_IMPORTS_AVAILABLE:
            self.optimizer = DatabaseOptimizer(self.test_db.name)
    
    def tearDown(self):
        """Clean up test environment"""
        try:
            os.unlink(self.test_db.name)
        except:
            pass
    
    @unittest.skipUnless(PHASE4_IMPORTS_AVAILABLE, "Phase 4 modules not available")
    def test_query_analysis(self):
        """Test query performance analysis"""
        # Simulate slow query
        query = "SELECT u.name, COUNT(o.id) FROM users u LEFT JOIN orders o ON u.id = o.user_id GROUP BY u.id"
        
        start_time = time.time()
        with sqlite3.connect(self.test_db.name) as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            results = cursor.fetchall()
        execution_time = time.time() - start_time
        
        # Analyze query
        profile = self.optimizer.analyze_query_execution(query, execution_time, len(results))
        
        self.assertIsNotNone(profile)
        self.assertGreater(profile.execution_time, 0)
        self.assertGreater(profile.complexity_score, 0)
        self.assertEqual(profile.frequency, 1)
    
    @unittest.skipUnless(PHASE4_IMPORTS_AVAILABLE, "Phase 4 modules not available")
    def test_index_recommendations(self):
        """Test index recommendation generation"""
        # Simulate queries that would benefit from indexes
        queries = [
            "SELECT * FROM users WHERE email = 'user_100@test.com'",
            "SELECT * FROM orders WHERE user_id = 500",
            "SELECT * FROM orders WHERE order_date > '2024-01-01' ORDER BY order_date",
            "SELECT COUNT(*) FROM orders WHERE amount > 50.0"
        ]
        
        # Analyze queries
        for query in queries:
            start_time = time.time()
            with sqlite3.connect(self.test_db.name) as conn:
                cursor = conn.cursor()
                cursor.execute(query)
                results = cursor.fetchall()
            execution_time = time.time() - start_time
            
            self.optimizer.analyze_query_execution(query, execution_time, len(results))
        
        # Generate recommendations
        recommendations = self.optimizer.generate_index_recommendations()
        
        self.assertIsInstance(recommendations, list)
        # Should have some recommendations for the queries above
        if recommendations:
            self.assertIn('table_name', recommendations[0].__dict__)
            self.assertIn('columns', recommendations[0].__dict__)
            self.assertIn('estimated_benefit', recommendations[0].__dict__)
    
    @unittest.skipUnless(PHASE4_IMPORTS_AVAILABLE, "Phase 4 modules not available")
    def test_performance_metrics(self):
        """Test performance metrics collection"""
        # Run several queries to generate metrics
        queries = [
            "SELECT COUNT(*) FROM users",
            "SELECT COUNT(*) FROM orders",
            "SELECT AVG(amount) FROM orders"
        ]
        
        for query in queries:
            execution_time = 0.1  # Simulated execution time
            result_count = 1
            self.optimizer.analyze_query_execution(query, execution_time, result_count)
        
        # Get performance metrics
        metrics = self.optimizer.get_performance_metrics()
        
        self.assertIsNotNone(metrics)
        self.assertGreaterEqual(metrics.avg_query_time, 0)
        self.assertIsInstance(metrics.slowest_queries, list)
        self.assertIsInstance(metrics.most_frequent_queries, list)

class TestPerformanceMonitor(unittest.TestCase):
    """Test performance monitoring system"""
    
    def setUp(self):
        """Set up test environment"""
        if PHASE4_IMPORTS_AVAILABLE:
            self.monitor = PerformanceMonitor()
    
    @unittest.skipUnless(PHASE4_IMPORTS_AVAILABLE, "Phase 4 modules not available")
    def test_query_performance_tracking(self):
        """Test query performance tracking"""
        # Track several queries
        queries = [
            ("SELECT * FROM test_table", 0.5, 100),
            ("SELECT COUNT(*) FROM test_table", 0.1, 1),
            ("SELECT * FROM test_table WHERE id > 50", 1.2, 50)
        ]
        
        for query, exec_time, row_count in queries:
            metric = self.monitor.track_query_performance(
                query, exec_time, rows_processed=row_count
            )
            
            self.assertIsNotNone(metric)
            self.assertEqual(metric.execution_time, exec_time)
            self.assertEqual(metric.rows_processed, row_count)
            self.assertGreaterEqual(metric.optimization_score, 0)
            self.assertLessEqual(metric.optimization_score, 1.0)
    
    @unittest.skipUnless(PHASE4_IMPORTS_AVAILABLE, "Phase 4 modules not available")
    def test_error_recording(self):
        """Test error recording and tracking"""
        # Record some errors
        self.monitor.record_error("SQL_ERROR", "Table not found", "SELECT * FROM missing_table")
        self.monitor.record_error("TIMEOUT", "Query timeout", "SELECT * FROM large_table")
        
        self.assertEqual(self.monitor.error_count, 2)
        self.assertGreater(len(self.monitor.performance_alerts), 0)
    
    @unittest.skipUnless(PHASE4_IMPORTS_AVAILABLE, "Phase 4 modules not available")
    def test_performance_baselines(self):
        """Test performance baseline establishment"""
        # Generate enough data for baselines
        for i in range(20):
            query = f"SELECT * FROM table_{i % 5}"
            exec_time = 0.1 + (i % 10) * 0.05  # Variable execution times
            self.monitor.track_query_performance(query, exec_time, rows_processed=i*10)
        
        # Establish baselines
        self.monitor.establish_performance_baselines()
        
        self.assertGreater(len(self.monitor.baselines), 0)
        
        # Check baseline properties
        for baseline_name, baseline in self.monitor.baselines.items():
            self.assertIsInstance(baseline.baseline_value, (int, float))
            self.assertIsInstance(baseline.acceptable_range, tuple)
            self.assertEqual(len(baseline.acceptable_range), 2)
    
    @unittest.skipUnless(PHASE4_IMPORTS_AVAILABLE, "Phase 4 modules not available")
    def test_dashboard_generation(self):
        """Test performance dashboard generation"""
        # Generate some performance data
        for i in range(10):
            query = f"SELECT * FROM test_table WHERE id = {i}"
            exec_time = 0.1 + i * 0.01
            self.monitor.track_query_performance(query, exec_time, rows_processed=1)
        
        # Get dashboard
        dashboard = self.monitor.get_performance_dashboard()
        
        self.assertIn('timestamp', dashboard)
        self.assertIn('summary', dashboard)
        self.assertIn('query_performance', dashboard)
        self.assertIn('system_performance', dashboard)
        
        # Check summary data
        summary = dashboard['summary']
        self.assertGreaterEqual(summary['total_queries'], 10)
        self.assertGreaterEqual(summary['queries_last_hour'], 0)

class TestMemoryManager(unittest.TestCase):
    """Test memory management system"""
    
    def setUp(self):
        """Set up test environment"""
        if PHASE4_IMPORTS_AVAILABLE:
            self.memory_manager = MemoryManager()
    
    @unittest.skipUnless(PHASE4_IMPORTS_AVAILABLE, "Phase 4 modules not available")
    def test_memory_pool_operations(self):
        """Test memory pool creation and allocation"""
        # Create memory pool
        pool = self.memory_manager.create_memory_pool("test_pool", 10, 1024)
        
        self.assertEqual(pool.name, "test_pool")
        self.assertEqual(pool.pool_size, 10)
        self.assertEqual(pool.block_size, 1024)
        self.assertEqual(pool.free_blocks, 10)
        
        # Allocate blocks
        block_ids = []
        for i in range(5):
            block_id = self.memory_manager.allocate_from_pool("test_pool")
            self.assertIsNotNone(block_id)
            block_ids.append(block_id)
        
        # Check pool state
        self.assertEqual(pool.allocated_blocks, 5)
        self.assertEqual(pool.free_blocks, 5)
        
        # Deallocate blocks
        for block_id in block_ids:
            self.memory_manager.deallocate_from_pool("test_pool", block_id)
        
        # Check pool state after deallocation
        self.assertEqual(pool.allocated_blocks, 0)
        self.assertEqual(pool.free_blocks, 10)
    
    @unittest.skipUnless(PHASE4_IMPORTS_AVAILABLE, "Phase 4 modules not available")
    def test_memory_optimization(self):
        """Test memory optimization functions"""
        # Perform optimization
        results = self.memory_manager.optimize_memory_usage()
        
        self.assertIsInstance(results, dict)
        self.assertIn('gc_collected', results)
        
        # Test emergency cleanup
        emergency_results = self.memory_manager.emergency_cleanup()
        
        self.assertIsInstance(emergency_results, dict)
        self.assertIn('emergency_gc_collected', emergency_results)
    
    @unittest.skipUnless(PHASE4_IMPORTS_AVAILABLE, "Phase 4 modules not available")
    def test_memory_statistics(self):
        """Test memory statistics collection"""
        # Create some memory pools for statistics
        self.memory_manager.create_memory_pool("stats_test", 20, 512)
        
        # Get statistics
        stats = self.memory_manager.get_memory_statistics()
        
        self.assertIn('timestamp', stats)
        self.assertIn('current_usage', stats)
        self.assertIn('memory_pools', stats)
        self.assertIn('gc_statistics', stats)
        
        # Check current usage data
        usage = stats['current_usage']
        self.assertIn('total_memory_mb', usage)
        self.assertIn('available_memory_mb', usage)
        self.assertIn('process_memory_mb', usage)

class TestScalabilityManager(unittest.TestCase):
    """Test scalability management system"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.test_db.close()
        
        # Create simple test database
        with sqlite3.connect(self.test_db.name) as conn:
            conn.execute('''
                CREATE TABLE scalability_test (
                    id INTEGER PRIMARY KEY,
                    data TEXT,
                    created_at TEXT
                )
            ''')
            
            for i in range(50):
                conn.execute(
                    "INSERT INTO scalability_test (data, created_at) VALUES (?, ?)",
                    (f"test_data_{i}", datetime.now().isoformat())
                )
        
        if PHASE4_IMPORTS_AVAILABLE:
            self.scalability_manager = ScalabilityManager(self.test_db.name)
    
    def tearDown(self):
        """Clean up test environment"""
        if PHASE4_IMPORTS_AVAILABLE:
            self.scalability_manager.shutdown()
        try:
            os.unlink(self.test_db.name)
        except:
            pass
    
    @unittest.skipUnless(PHASE4_IMPORTS_AVAILABLE, "Phase 4 modules not available")
    def test_connection_pool(self):
        """Test database connection pooling"""
        # Test getting connections
        connections_used = []
        for i in range(3):
            with self.scalability_manager.get_database_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM scalability_test")
                result = cursor.fetchone()
                self.assertIsNotNone(result)
                connections_used.append(conn)
        
        # All connections should be properly returned to pool
        self.assertLessEqual(
            self.scalability_manager.connection_pool.active_connections,
            self.scalability_manager.connection_config.max_connections
        )
    
    @unittest.skipUnless(PHASE4_IMPORTS_AVAILABLE, "Phase 4 modules not available")
    def test_task_submission_and_processing(self):
        """Test task submission and processing"""
        # Submit tasks
        task_ids = []
        for i in range(5):
            task_id = self.scalability_manager.submit_task(
                'sql_query',
                {
                    'query': 'SELECT COUNT(*) FROM scalability_test',
                    'db_path': self.test_db.name
                },
                priority=5
            )
            task_ids.append(task_id)
        
        self.assertEqual(len(task_ids), 5)
        
        # Process tasks
        results = self.scalability_manager.process_tasks(max_concurrent=2)
        
        self.assertIsInstance(results, dict)
        self.assertIn('completed', results)
        self.assertIn('failed', results)
        self.assertGreaterEqual(results['completed'], 0)
    
    @unittest.skipUnless(PHASE4_IMPORTS_AVAILABLE, "Phase 4 modules not available")
    def test_worker_management(self):
        """Test worker node management"""
        initial_workers = len(self.scalability_manager.worker_manager.workers)
        
        # Create additional worker
        worker_id = self.scalability_manager.worker_manager.create_worker('thread', capacity=3)
        self.assertIsNotNone(worker_id)
        
        new_workers = len(self.scalability_manager.worker_manager.workers)
        self.assertEqual(new_workers, initial_workers + 1)
        
        # Get available worker
        worker = self.scalability_manager.worker_manager.get_available_worker()
        self.assertIsNotNone(worker)
        self.assertIn(worker.status, ['idle', 'active'])
    
    @unittest.skipUnless(PHASE4_IMPORTS_AVAILABLE, "Phase 4 modules not available")
    def test_system_status(self):
        """Test system status reporting"""
        status = self.scalability_manager.get_system_status()
        
        self.assertIn('timestamp', status)
        self.assertIn('system_metrics', status)
        self.assertIn('connection_pool', status)
        self.assertIn('task_queue', status)
        self.assertIn('workers', status)
        self.assertIn('auto_scaling', status)
        
        # Check worker statistics
        workers = status['workers']
        self.assertIn('total_workers', workers)
        self.assertIn('active_workers', workers)
        self.assertIn('idle_workers', workers)

class TestPhase4Integration(unittest.TestCase):
    """Test integration between Phase 4 components and previous phases"""
    
    def setUp(self):
        """Set up integrated test environment"""
        self.test_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.test_db.close()
        
        # Create comprehensive test database
        with sqlite3.connect(self.test_db.name) as conn:
            # Create tables that would be analyzed by previous phases
            conn.execute('''
                CREATE TABLE customers (
                    customer_id INTEGER PRIMARY KEY,
                    email_address TEXT,
                    full_name TEXT,
                    registration_date TEXT,
                    country_code TEXT
                )
            ''')
            
            conn.execute('''
                CREATE TABLE products (
                    product_id INTEGER PRIMARY KEY,
                    product_name TEXT,
                    category_name TEXT,
                    unit_price REAL,
                    manufacturer_name TEXT
                )
            ''')
            
            conn.execute('''
                CREATE TABLE transactions (
                    transaction_id INTEGER PRIMARY KEY,
                    customer_id INTEGER,
                    product_id INTEGER,
                    quantity INTEGER,
                    total_amount REAL,
                    transaction_date TEXT,
                    order_status TEXT,
                    FOREIGN KEY (customer_id) REFERENCES customers(customer_id),
                    FOREIGN KEY (product_id) REFERENCES products(product_id)
                )
            ''')
            
            # Insert test data
            import random
            
            # Customers
            countries = ['US', 'CA', 'UK', 'DE', 'FR']
            for i in range(100):
                conn.execute('''
                    INSERT INTO customers (email_address, full_name, registration_date, country_code)
                    VALUES (?, ?, ?, ?)
                ''', (
                    f"customer_{i}@example.com",
                    f"Customer {i} Name",
                    (datetime.now() - timedelta(days=random.randint(1, 365))).isoformat(),
                    random.choice(countries)
                ))
            
            # Products
            categories = ['Electronics', 'Clothing', 'Books', 'Home', 'Sports']
            manufacturers = ['Brand A', 'Brand B', 'Brand C', 'Brand D', 'Brand E']
            for i in range(50):
                conn.execute('''
                    INSERT INTO products (product_name, category_name, unit_price, manufacturer_name)
                    VALUES (?, ?, ?, ?)
                ''', (
                    f"Product {i}",
                    random.choice(categories),
                    random.uniform(10.0, 500.0),
                    random.choice(manufacturers)
                ))
            
            # Transactions
            statuses = ['completed', 'pending', 'cancelled', 'refunded']
            for i in range(500):
                conn.execute('''
                    INSERT INTO transactions (customer_id, product_id, quantity, total_amount, transaction_date, order_status)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    random.randint(1, 100),
                    random.randint(1, 50),
                    random.randint(1, 5),
                    random.uniform(10.0, 1000.0),
                    (datetime.now() - timedelta(days=random.randint(1, 180))).isoformat(),
                    random.choice(statuses)
                ))
    
    def tearDown(self):
        """Clean up test environment"""
        try:
            os.unlink(self.test_db.name)
        except:
            pass
    
    @unittest.skipUnless(PHASE4_IMPORTS_AVAILABLE, "Phase 4 modules not available")
    def test_integrated_query_processing(self):
        """Test integrated query processing with all phases"""
        # Initialize Phase 4 components
        cache = get_intelligent_cache()
        optimizer = get_database_optimizer(self.test_db.name)
        monitor = get_performance_monitor(self.test_db.name)
        scalability = get_scalability_manager(self.test_db.name)
        
        # Test query that should trigger multiple optimizations
        test_query = "SELECT c.full_name, SUM(t.total_amount) FROM customers c JOIN transactions t ON c.customer_id = t.customer_id WHERE t.order_status = 'completed' GROUP BY c.customer_id ORDER BY SUM(t.total_amount) DESC LIMIT 10"
        
        # Execute query and measure performance
        start_time = time.time()
        results = scalability.execute_query(test_query)
        execution_time = time.time() - start_time
        
        # Track performance
        monitor.track_query_performance(test_query, execution_time, len(results))
        
        # Analyze for optimization opportunities
        profile = optimizer.analyze_query_execution(test_query, execution_time, len(results))
        
        # Cache results
        cache_key = cache._generate_semantic_cache_key(test_query, {"tables": ["customers", "transactions"]})
        cache.set(cache_key, {"results": results, "execution_time": execution_time}, category="query_results")
        
        # Verify integration worked
        self.assertIsNotNone(results)
        self.assertGreater(len(results), 0)
        self.assertIsNotNone(profile)
        
        # Test cache retrieval
        cached_results = cache.get(cache_key, category="query_results")
        self.assertIsNotNone(cached_results)
        self.assertEqual(len(cached_results["results"]), len(results))
    
    @unittest.skipUnless(PHASE4_IMPORTS_AVAILABLE, "Phase 4 modules not available")
    def test_performance_optimization_pipeline(self):
        """Test complete performance optimization pipeline"""
        # Initialize all Phase 4 components
        cache = get_intelligent_cache()
        optimizer = get_database_optimizer(self.test_db.name)
        monitor = get_performance_monitor(self.test_db.name)
        memory_manager = get_memory_manager()
        scalability = get_scalability_manager(self.test_db.name)
        
        # Simulate workload that triggers optimizations
        queries = [
            "SELECT * FROM customers WHERE email_address LIKE '%@example.com'",
            "SELECT COUNT(*) FROM transactions WHERE order_status = 'completed'",
            "SELECT p.category_name, AVG(p.unit_price) FROM products p GROUP BY p.category_name",
            "SELECT c.country_code, COUNT(*) FROM customers c GROUP BY c.country_code",
            "SELECT * FROM transactions t JOIN products p ON t.product_id = p.product_id WHERE p.unit_price > 100"
        ]
        
        # Execute queries multiple times to build performance data
        for iteration in range(3):
            for query in queries:
                # Check cache first
                cache_key = cache._generate_semantic_cache_key(query, {"db": self.test_db.name})
                cached_result = cache.get(cache_key, category="query_results")
                
                if cached_result:
                    # Use cached result
                    results = cached_result["results"]
                    execution_time = 0.001  # Cache hit time
                else:
                    # Execute query
                    start_time = time.time()
                    results = scalability.execute_query(query)
                    execution_time = time.time() - start_time
                    
                    # Cache result
                    cache.set(cache_key, {"results": results, "query": query}, category="query_results")
                
                # Track performance
                monitor.track_query_performance(query, execution_time, len(results))
                
                # Analyze for optimization
                optimizer.analyze_query_execution(query, execution_time, len(results))
        
        # Generate optimization recommendations
        recommendations = optimizer.generate_index_recommendations()
        
        # Get performance dashboard
        dashboard = monitor.get_performance_dashboard()
        
        # Check memory usage
        memory_stats = memory_manager.get_memory_statistics()
        
        # Get system status
        system_status = scalability.get_system_status()
        
        # Verify optimization pipeline worked
        self.assertIsInstance(recommendations, list)
        self.assertIn('summary', dashboard)
        self.assertIn('current_usage', memory_stats)
        self.assertIn('system_metrics', system_status)
        
        # Cache should have improved hit rate
        cache_metrics = cache.get_performance_metrics()
        if cache_metrics['total_requests'] > 0:
            self.assertGreater(cache_metrics['hit_rate'], 0)
    
    @unittest.skipUnless(PHASE4_IMPORTS_AVAILABLE, "Phase 4 modules not available")
    def test_scalability_under_load(self):
        """Test system scalability under concurrent load"""
        scalability = get_scalability_manager(self.test_db.name)
        monitor = get_performance_monitor(self.test_db.name)
        
        # Submit multiple concurrent tasks
        task_ids = []
        queries = [
            "SELECT COUNT(*) FROM customers",
            "SELECT COUNT(*) FROM products", 
            "SELECT COUNT(*) FROM transactions",
            "SELECT AVG(unit_price) FROM products",
            "SELECT COUNT(DISTINCT country_code) FROM customers"
        ]
        
        # Submit tasks
        for i in range(20):  # 20 tasks total
            query = queries[i % len(queries)]
            task_id = scalability.submit_task(
                'sql_query',
                {'query': query, 'db_path': self.test_db.name},
                priority=random.randint(1, 10)
            )
            task_ids.append(task_id)
        
        # Process tasks concurrently
        start_time = time.time()
        results = scalability.process_tasks(max_concurrent=5)
        total_time = time.time() - start_time
        
        # Verify results
        self.assertEqual(len(task_ids), 20)
        self.assertGreaterEqual(results['completed'], 15)  # Allow for some variance
        self.assertLess(total_time, 30)  # Should complete within 30 seconds
        
        # Check system remained stable
        status = scalability.get_system_status()
        self.assertGreater(status['workers']['total_workers'], 0)

def run_phase4_performance_benchmark():
    """Run performance benchmark for Phase 4 components"""
    if not PHASE4_IMPORTS_AVAILABLE:
        print("Phase 4 modules not available for benchmarking")
        return
    
    print("\n" + "="*60)
    print("PHASE 4 PERFORMANCE BENCHMARK")
    print("="*60)
    
    # Create test database
    test_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
    test_db.close()
    
    try:
        # Setup benchmark database
        with sqlite3.connect(test_db.name) as conn:
            conn.execute('''
                CREATE TABLE benchmark_data (
                    id INTEGER PRIMARY KEY,
                    category TEXT,
                    value REAL,
                    description TEXT,
                    created_at TEXT
                )
            ''')
            
            # Insert 10,000 rows for benchmarking
            print("Setting up benchmark database with 10,000 rows...")
            import random
            categories = ['A', 'B', 'C', 'D', 'E']
            for i in range(10000):
                conn.execute('''
                    INSERT INTO benchmark_data (category, value, description, created_at)
                    VALUES (?, ?, ?, ?)
                ''', (
                    random.choice(categories),
                    random.uniform(1.0, 1000.0),
                    f"Benchmark data row {i}",
                    datetime.now().isoformat()
                ))
        
        # Initialize components
        cache = IntelligentCache()
        optimizer = DatabaseOptimizer(test_db.name)
        monitor = PerformanceMonitor(test_db.name)
        scalability = ScalabilityManager(test_db.name)
        
        # Benchmark queries
        benchmark_queries = [
            "SELECT * FROM benchmark_data WHERE category = 'A'",
            "SELECT category, COUNT(*) FROM benchmark_data GROUP BY category",
            "SELECT * FROM benchmark_data WHERE value > 500 ORDER BY value DESC LIMIT 100",
            "SELECT AVG(value), MIN(value), MAX(value) FROM benchmark_data",
            "SELECT * FROM benchmark_data WHERE description LIKE '%500%'"
        ]
        
        print("Running benchmark queries...")
        
        # Benchmark 1: Raw query performance
        print("\n1. Raw Query Performance:")
        raw_times = []
        for query in benchmark_queries:
            start_time = time.time()
            results = scalability.execute_query(query)
            end_time = time.time()
            execution_time = end_time - start_time
            raw_times.append(execution_time)
            
            print(f"   Query: {query[:50]}...")
            print(f"   Time: {execution_time:.4f}s, Rows: {len(results)}")
            
            # Track with monitor
            monitor.track_query_performance(query, execution_time, len(results))
        
        print(f"   Average raw query time: {sum(raw_times)/len(raw_times):.4f}s")
        
        # Benchmark 2: Cached query performance
        print("\n2. Cached Query Performance:")
        cached_times = []
        cache_hits = 0
        
        for query in benchmark_queries:
            cache_key = cache._generate_semantic_cache_key(query, {"db": test_db.name})
            
            start_time = time.time()
            cached_result = cache.get(cache_key, category="query_results")
            
            if cached_result:
                # Cache hit
                results = cached_result.get("results", [])
                cache_hits += 1
                cache_time = 0.001  # Simulate cache retrieval time
            else:
                # Cache miss - execute and cache
                results = scalability.execute_query(query)
                cache.set(cache_key, {"results": results}, category="query_results")
                cache_time = time.time() - start_time
            
            cached_times.append(cache_time)
            print(f"   Query: {query[:50]}...")
            print(f"   Time: {cache_time:.4f}s, Cache: {'HIT' if cached_result else 'MISS'}")
        
        print(f"   Average cached query time: {sum(cached_times)/len(cached_times):.4f}s")
        print(f"   Cache hit rate: {cache_hits}/{len(benchmark_queries)}")
        
        # Benchmark 3: Concurrent processing
        print("\n3. Concurrent Processing Performance:")
        
        # Submit multiple tasks
        task_ids = []
        for i in range(50):  # 50 concurrent tasks
            query = benchmark_queries[i % len(benchmark_queries)]
            task_id = scalability.submit_task(
                'sql_query',
                {'query': query, 'db_path': test_db.name},
                priority=5
            )
            task_ids.append(task_id)
        
        # Process concurrently
        concurrent_start = time.time()
        concurrent_results = scalability.process_tasks(max_concurrent=10)
        concurrent_end = time.time()
        concurrent_time = concurrent_end - concurrent_start
        
        print(f"   Total tasks: 50")
        print(f"   Completed: {concurrent_results['completed']}")
        print(f"   Failed: {concurrent_results['failed']}")
        print(f"   Total time: {concurrent_time:.4f}s")
        print(f"   Tasks per second: {concurrent_results['completed']/concurrent_time:.2f}")
        
        # Benchmark 4: Memory usage
        print("\n4. Memory Usage Analysis:")
        memory_manager = get_memory_manager()
        memory_stats = memory_manager.get_memory_statistics()
        
        print(f"   Process memory: {memory_stats['current_usage']['process_memory_mb']:.2f} MB")
        print(f"   Available memory: {memory_stats['current_usage']['available_memory_mb']:.2f} MB")
        print(f"   Memory pools: {len(memory_stats['memory_pools'])}")
        
        # Generate performance reports
        print("\n5. Generating Performance Reports:")
        
        # Optimization recommendations
        recommendations = optimizer.generate_index_recommendations()
        print(f"   Index recommendations: {len(recommendations)}")
        
        # Performance dashboard
        dashboard = monitor.get_performance_dashboard()
        print(f"   Queries tracked: {dashboard['summary']['total_queries']}")
        print(f"   Average execution time: {dashboard['query_performance']['avg_execution_time']:.4f}s")
        
        # System status
        system_status = scalability.get_system_status()
        print(f"   Active workers: {system_status['workers']['active_workers']}")
        print(f"   Queue size: {system_status['task_queue']['current_queue_size']}")
        
        print("\n" + "="*60)
        print("BENCHMARK COMPLETED SUCCESSFULLY")
        print("="*60)
        
        # Cleanup
        scalability.shutdown()
        
    except Exception as e:
        print(f"Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        try:
            os.unlink(test_db.name)
        except:
            pass

def main():
    """Main test runner"""
    print("PHERS Phase 4 Integration Testing")
    print("=" * 50)
    
    if not PHASE4_IMPORTS_AVAILABLE:
        print("ERROR: Phase 4 modules are not available for testing")
        print("Please ensure all Phase 4 modules are properly installed")
        return 1
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestIntelligentCache,
        TestDatabaseOptimizer,
        TestPerformanceMonitor,
        TestMemoryManager,
        TestScalabilityManager,
        TestPhase4Integration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "=" * 50)
    print("PHASE 4 TEST SUMMARY")
    print("=" * 50)
    print(f"Tests Run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success Rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    # Run performance benchmark
    run_phase4_performance_benchmark()
    
    # Log test results
    try:
        with open('/data/activity.log', 'a', encoding='utf-8') as f:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "module": "TestPhase4Integration",
                "activity": "Phase 4 integration tests completed",
                "details": {
                    "tests_run": result.testsRun,
                    "failures": len(result.failures),
                    "errors": len(result.errors),
                    "success_rate": (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
                }
            }
            f.write(json.dumps(log_entry) + '\n')
    except Exception as e:
        print(f"Warning: Could not log results: {e}")
    
    return 0 if (len(result.failures) == 0 and len(result.errors) == 0) else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)