"""
Simplified Phase 4 Integration Tests for PHERS
Performance Optimization Module Testing (No External Dependencies)

This module provides basic testing for Phase 4 components without requiring
external dependencies like psutil.
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

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from intelligent_cache import IntelligentCache, get_intelligent_cache
    INTELLIGENT_CACHE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: intelligent_cache not available: {e}")
    INTELLIGENT_CACHE_AVAILABLE = False

try:
    from database_optimizer import DatabaseOptimizer, get_database_optimizer
    DATABASE_OPTIMIZER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: database_optimizer not available: {e}")
    DATABASE_OPTIMIZER_AVAILABLE = False

class TestIntelligentCacheSimplified(unittest.TestCase):
    """Simplified test for intelligent caching system"""
    
    def setUp(self):
        """Set up test environment"""
        if INTELLIGENT_CACHE_AVAILABLE:
            self.cache = IntelligentCache()
    
    @unittest.skipUnless(INTELLIGENT_CACHE_AVAILABLE, "IntelligentCache not available")
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
    
    @unittest.skipUnless(INTELLIGENT_CACHE_AVAILABLE, "IntelligentCache not available")
    def test_cache_basic_functionality(self):
        """Test basic cache functionality without expiration"""
        # Test multiple cache operations
        test_data = [
            ("key1", {"data": "value1"}),
            ("key2", {"data": "value2", "count": 42}),
            ("key3", {"results": [1, 2, 3, 4, 5]})
        ]
        
        # Set all values
        for key, value in test_data:
            success = self.cache.set(key, value, category="query_results")
            self.assertTrue(success)
        
        # Verify all values can be retrieved
        for key, expected_value in test_data:
            cached_value = self.cache.get(key, category="query_results")
            self.assertIsNotNone(cached_value)
            self.assertEqual(cached_value, expected_value)
        
        # Test cache statistics
        stats = self.cache.get_cache_stats()
        self.assertGreaterEqual(stats["total_entries"], 3)
    
    @unittest.skipUnless(INTELLIGENT_CACHE_AVAILABLE, "IntelligentCache not available")
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
        metrics = self.cache.get_cache_stats()  # Use get_cache_stats instead
        self.assertGreater(metrics["total_hits"], 0)
        self.assertGreater(metrics["total_misses"], 0)
        self.assertGreater(metrics["hit_rate"], 0)

class TestDatabaseOptimizerSimplified(unittest.TestCase):
    """Simplified test for database optimization system"""
    
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
            for i in range(100):
                conn.execute(
                    "INSERT INTO users (email, name, created_at) VALUES (?, ?, ?)",
                    (f"user_{i}@test.com", f"User {i}", datetime.now().isoformat())
                )
            
            for i in range(500):
                conn.execute(
                    "INSERT INTO orders (user_id, product_name, amount, order_date) VALUES (?, ?, ?, ?)",
                    (i % 100 + 1, f"Product {i % 20}", (i % 50) * 5.99, datetime.now().isoformat())
                )
        
        if DATABASE_OPTIMIZER_AVAILABLE:
            # Create a simplified activity log path for testing
            self.activity_log_path = tempfile.mktemp(suffix='.log')
            self.optimizer = DatabaseOptimizer(self.test_db.name, self.activity_log_path)
    
    def tearDown(self):
        """Clean up test environment"""
        try:
            os.unlink(self.test_db.name)
        except:
            pass
        
        if hasattr(self, 'activity_log_path'):
            try:
                os.unlink(self.activity_log_path)
            except:
                pass
    
    @unittest.skipUnless(DATABASE_OPTIMIZER_AVAILABLE, "DatabaseOptimizer not available")
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
    
    @unittest.skipUnless(DATABASE_OPTIMIZER_AVAILABLE, "DatabaseOptimizer not available")
    def test_index_recommendations(self):
        """Test index recommendation generation"""
        # Simulate queries that would benefit from indexes
        queries = [
            "SELECT * FROM users WHERE email = 'user_10@test.com'",
            "SELECT * FROM orders WHERE user_id = 50",
            "SELECT * FROM orders WHERE order_date > '2024-01-01' ORDER BY order_date",
            "SELECT COUNT(*) FROM orders WHERE amount > 25.0"
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

class TestPhase4BasicIntegration(unittest.TestCase):
    """Test basic integration between available Phase 4 components"""
    
    def setUp(self):
        """Set up integrated test environment"""
        self.test_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.test_db.close()
        
        # Create simple test database
        with sqlite3.connect(self.test_db.name) as conn:
            conn.execute('''
                CREATE TABLE test_data (
                    id INTEGER PRIMARY KEY,
                    name TEXT,
                    value INTEGER,
                    category TEXT
                )
            ''')
            
            # Insert test data
            for i in range(50):
                conn.execute(
                    "INSERT INTO test_data (name, value, category) VALUES (?, ?, ?)",
                    (f"item_{i}", i * 10, f"cat_{i % 5}")
                )
    
    def tearDown(self):
        """Clean up test environment"""
        try:
            os.unlink(self.test_db.name)
        except:
            pass
    
    @unittest.skipUnless(INTELLIGENT_CACHE_AVAILABLE and DATABASE_OPTIMIZER_AVAILABLE, "Phase 4 modules not available")
    def test_cache_and_optimizer_integration(self):
        """Test integration between cache and optimizer"""
        # Initialize components
        cache = get_intelligent_cache()
        activity_log_path = tempfile.mktemp(suffix='.log')
        optimizer = DatabaseOptimizer(self.test_db.name, activity_log_path)
        
        # Test query
        test_query = "SELECT category, COUNT(*) FROM test_data GROUP BY category"
        
        # Execute query and measure performance
        start_time = time.time()
        with sqlite3.connect(self.test_db.name) as conn:
            cursor = conn.cursor()
            cursor.execute(test_query)
            results = cursor.fetchall()
        execution_time = time.time() - start_time
        
        # Analyze query with optimizer
        profile = optimizer.analyze_query_execution(test_query, execution_time, len(results))
        self.assertIsNotNone(profile)
        
        # Cache the results using a simple key
        cache_key = f"query_{hash(test_query)}"
        cache_success = cache.set(cache_key, {"results": results, "execution_time": execution_time}, category="query_results")
        self.assertTrue(cache_success)
        
        # Retrieve from cache
        cached_results = cache.get(cache_key, category="query_results")
        self.assertIsNotNone(cached_results)
        self.assertEqual(len(cached_results["results"]), len(results))
        
        # Clean up
        try:
            os.unlink(activity_log_path)
        except:
            pass

def run_basic_performance_test():
    """Run a basic performance test for available Phase 4 components"""
    print("\n" + "="*50)
    print("BASIC PHASE 4 PERFORMANCE TEST")
    print("="*50)
    
    if not (INTELLIGENT_CACHE_AVAILABLE or DATABASE_OPTIMIZER_AVAILABLE):
        print("No Phase 4 modules available for testing")
        return
    
    # Create test database
    test_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
    test_db.close()
    
    try:
        # Setup test database
        with sqlite3.connect(test_db.name) as conn:
            conn.execute('''
                CREATE TABLE performance_test (
                    id INTEGER PRIMARY KEY,
                    category TEXT,
                    value REAL,
                    description TEXT
                )
            ''')
            
            # Insert 1,000 rows for testing
            print("Setting up test database with 1,000 rows...")
            for i in range(1000):
                conn.execute('''
                    INSERT INTO performance_test (category, value, description)
                    VALUES (?, ?, ?)
                ''', (
                    f"cat_{i % 10}",
                    float(i),
                    f"Test data row {i}"
                ))
        
        # Test queries
        test_queries = [
            "SELECT COUNT(*) FROM performance_test",
            "SELECT category, COUNT(*) FROM performance_test GROUP BY category",
            "SELECT * FROM performance_test WHERE value > 500",
            "SELECT AVG(value) FROM performance_test WHERE category = 'cat_1'"
        ]
        
        if INTELLIGENT_CACHE_AVAILABLE:
            print("\n1. Testing Intelligent Cache:")
            cache = IntelligentCache()
            
            # Test cache performance
            cache_times = []
            for query in test_queries:
                cache_key = f"query_{hash(query)}"
                
                # Execute query first time (cache miss)
                start_time = time.time()
                with sqlite3.connect(test_db.name) as conn:
                    cursor = conn.cursor()
                    cursor.execute(query)
                    results = cursor.fetchall()
                miss_time = time.time() - start_time
                
                # Cache the results
                cache.set(cache_key, {"results": results}, category="query_results")
                
                # Test cache retrieval
                start_time = time.time()
                cached_result = cache.get(cache_key, category="query_results")
                hit_time = time.time() - start_time
                
                cache_times.append((miss_time, hit_time))
                print(f"   Query: {query[:40]}...")
                print(f"   Miss time: {miss_time:.4f}s, Hit time: {hit_time:.6f}s")
                print(f"   Speedup: {miss_time/hit_time:.1f}x")
            
            # Cache performance metrics
            metrics = cache.get_cache_stats()
            print(f"   Cache hit rate: {metrics['hit_rate']:.2%}")
            print(f"   Total entries: {metrics['total_entries']}")
        
        if DATABASE_OPTIMIZER_AVAILABLE:
            print("\n2. Testing Database Optimizer:")
            activity_log_path = tempfile.mktemp(suffix='.log')
            optimizer = DatabaseOptimizer(test_db.name, activity_log_path)
            
            # Analyze queries
            for query in test_queries:
                start_time = time.time()
                with sqlite3.connect(test_db.name) as conn:
                    cursor = conn.cursor()
                    cursor.execute(query)
                    results = cursor.fetchall()
                execution_time = time.time() - start_time
                
                profile = optimizer.analyze_query_execution(query, execution_time, len(results))
                print(f"   Query: {query[:40]}...")
                print(f"   Time: {execution_time:.4f}s, Complexity: {profile.complexity_score:.2f}")
            
            # Generate recommendations
            recommendations = optimizer.generate_index_recommendations()
            print(f"   Index recommendations generated: {len(recommendations)}")
            
            # Clean up
            try:
                os.unlink(activity_log_path)
            except:
                pass
        
        print("\n" + "="*50)
        print("BASIC PERFORMANCE TEST COMPLETED")
        print("="*50)
    
    except Exception as e:
        print(f"Performance test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        try:
            os.unlink(test_db.name)
        except:
            pass

def main():
    """Main test runner"""
    print("PHERS Phase 4 Simplified Integration Testing")
    print("=" * 50)
    
    available_modules = []
    if INTELLIGENT_CACHE_AVAILABLE:
        available_modules.append("IntelligentCache")
    if DATABASE_OPTIMIZER_AVAILABLE:
        available_modules.append("DatabaseOptimizer")
    
    if not available_modules:
        print("ERROR: No Phase 4 modules are available for testing")
        return 1
    
    print(f"Available modules: {', '.join(available_modules)}")
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases based on available modules
    if INTELLIGENT_CACHE_AVAILABLE:
        tests = unittest.TestLoader().loadTestsFromTestCase(TestIntelligentCacheSimplified)
        test_suite.addTests(tests)
    
    if DATABASE_OPTIMIZER_AVAILABLE:
        tests = unittest.TestLoader().loadTestsFromTestCase(TestDatabaseOptimizerSimplified)
        test_suite.addTests(tests)
    
    if INTELLIGENT_CACHE_AVAILABLE and DATABASE_OPTIMIZER_AVAILABLE:
        tests = unittest.TestLoader().loadTestsFromTestCase(TestPhase4BasicIntegration)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "=" * 50)
    print("PHASE 4 SIMPLIFIED TEST SUMMARY")
    print("=" * 50)
    print(f"Tests Run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    if result.testsRun > 0:
        print(f"Success Rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    # Run basic performance test
    run_basic_performance_test()
    
    # Log test results
    try:
        os.makedirs('data', exist_ok=True)
        with open('data/activity.log', 'a', encoding='utf-8') as f:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "module": "TestPhase4Simplified",
                "activity": "Phase 4 simplified integration tests completed",
                "details": {
                    "tests_run": result.testsRun,
                    "failures": len(result.failures),
                    "errors": len(result.errors),
                    "success_rate": (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100 if result.testsRun > 0 else 0,
                    "available_modules": available_modules
                }
            }
            f.write(json.dumps(log_entry) + '\n')
    except Exception as e:
        print(f"Warning: Could not log results: {e}")
    
    return 0 if (len(result.failures) == 0 and len(result.errors) == 0) else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)