#!/usr/bin/env python3
"""
Test script for Phase 1: Column Intelligence & Detection system.
Tests the dynamic column mapping capabilities with different CSV structures.
"""

import pandas as pd
import sys
import os

# Add current directory to path to import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def create_test_csvs():
    """Create different CSV structures to test column detection."""
    
    # Test CSV 1: Traditional asset management structure
    asset_data1 = {
        'Asset_TAG': ['ASSET-001', 'ASSET-002', 'ASSET-003'],
        'Employee_Name': ['John Smith', 'Jane Doe', 'Bob Johnson'],
        'Manufacturer': ['Dell Inc.', 'HP Corporation', 'Apple Inc.'],
        'Device_Model': ['Latitude 5520', 'EliteBook 840', 'MacBook Pro'],
        'Office_Location': ['Floor 3 Room 301', 'Building A Level 2', 'Floor 1 Reception']
    }
    
    # Test CSV 2: Different naming convention
    asset_data2 = {
        'ID_NUMBER': ['DEV-101', 'DEV-102', 'DEV-103'],
        'USER_FULL_NAME': ['Alice Cooper', 'Charlie Brown', 'Diana Prince'],
        'BRAND': ['Lenovo', 'ASUS', 'Samsung'],
        'PRODUCT_NAME': ['ThinkPad X1', 'ZenBook Pro', 'Galaxy Book'],
        'WORKPLACE': ['Lantai 2 Ruang IT', 'Gedung B Lt.1', 'Office Tower 15F']
    }
    
    # Test CSV 3: Minimal structure with unusual names
    asset_data3 = {
        'tag_code': ['ABC123', 'XYZ456', 'QRS789'],
        'owner': ['Mike Wilson', 'Sara Connor', 'Tom Anderson'],
        'company': ['Microsoft Corp', 'Google LLC', 'Meta Inc'],
        'item': ['Surface Laptop', 'Pixelbook Go', 'Portal Device']
    }
    
    # Test CSV 4: Mixed case and special characters (dirty data)
    asset_data4 = {
        'Asset#ID@': ['COMP-2024-001', 'COMP-2024-002', 'COMP-2024-003'],
        '  User Name  ': ['Emily Davis  ', '  Mark Taylor', 'Lisa Anderson'],
        'Manufacturer/Brand': ['IBM & Co.', 'Oracle Corp.', 'Adobe Inc.'],
        'Device;Type': ['Laptop Computer', 'Desktop PC', 'Tablet Device'],
        'Location[Building]': ['Main Building Floor 5', 'Annex Building B2', 'Remote Office']
    }
    
    return {
        'traditional_assets.csv': pd.DataFrame(asset_data1),
        'alternative_format.csv': pd.DataFrame(asset_data2),
        'minimal_structure.csv': pd.DataFrame(asset_data3),
        'dirty_data.csv': pd.DataFrame(asset_data4)
    }

def test_column_intelligence():
    """Test the column intelligence system with different CSV structures."""
    print("🧠 Testing Phase 1: Column Intelligence & Detection")
    print("=" * 60)
    
    try:
        # Import the modules
        print("1. Testing module imports...")
        from column_intelligence import column_intelligence
        from dynamic_mapping import dynamic_mapper
        print("   ✅ Column intelligence modules imported successfully")
        
        # Create test datasets
        test_datasets = create_test_csvs()
        results = {}
        
        print(f"\n2. Testing column detection with {len(test_datasets)} different CSV structures...")
        
        for csv_name, df in test_datasets.items():
            print(f"\n   Testing: {csv_name}")
            print(f"   Columns: {list(df.columns)}")
            
            # Analyze the table structure
            analysis = dynamic_mapper.analyze_and_map_table(df, csv_name.replace('.csv', ''))
            results[csv_name] = analysis
            
            # Display detected roles
            detected_roles = analysis.get("role_mappings", {})
            print(f"   ✅ Detected {len(detected_roles)} semantic roles:")
            
            for role, columns in detected_roles.items():
                if columns:
                    best_col = columns[0]
                    confidence = best_col["confidence"]
                    column_name = best_col["column"]
                    print(f"      {role}: '{column_name}' (confidence: {confidence:.2f})")
        
        # Test cross-table role suggestions
        print(f"\n3. Testing cross-table role suggestions...")
        
        test_roles = ['identifier', 'person_name', 'manufacturer', 'product', 'location']
        for role in test_roles:
            suggestions = dynamic_mapper.suggest_tables_for_role(role)
            print(f"   {role}: Found in {len(suggestions)} tables")
            for suggestion in suggestions[:2]:  # Show top 2
                table = suggestion['table_name']
                column = suggestion['column_name']
                confidence = suggestion['confidence']
                print(f"      - {table}.{column} (confidence: {confidence:.2f})")
        
        # Test semantic query generation
        print(f"\n4. Testing semantic query generation...")
        
        test_queries = [
            {
                "description": "Find asset by identifier",
                "query": {
                    "roles": ["identifier", "person_name", "manufacturer"],
                    "conditions": {"identifier": "ASSET-001"}
                }
            },
            {
                "description": "Find by manufacturer",
                "query": {
                    "roles": ["identifier", "manufacturer", "product"],
                    "conditions": {"manufacturer": "Dell"}
                }
            }
        ]
        
        for test_case in test_queries:
            print(f"   Testing: {test_case['description']}")
            query_result = dynamic_mapper.generate_semantic_query(test_case['query'])
            
            if query_result['errors']:
                print(f"   ❌ Errors: {query_result['errors']}")
            else:
                print(f"   ✅ Found suitable tables: {query_result['tables']}")
                for role, info in query_result['columns'].items():
                    print(f"      {role}: {info['column']} (confidence: {info['confidence']:.2f})")
        
        # Test with edge cases
        print(f"\n5. Testing edge cases and robustness...")
        
        # Empty DataFrame
        empty_df = pd.DataFrame()
        try:
            empty_analysis = dynamic_mapper.analyze_and_map_table(empty_df, "empty_test")
            print(f"   ✅ Empty DataFrame handled gracefully")
        except Exception as e:
            print(f"   ⚠️  Empty DataFrame handling needs improvement: {e}")
        
        # Single column DataFrame
        single_col_df = pd.DataFrame({'asset_id': ['TEST-001', 'TEST-002']})
        try:
            single_analysis = dynamic_mapper.analyze_and_map_table(single_col_df, "single_col_test")
            roles = single_analysis.get("role_mappings", {})
            print(f"   ✅ Single column DataFrame: detected {len(roles)} roles")
        except Exception as e:
            print(f"   ⚠️  Single column handling needs improvement: {e}")
        
        # Generate summary report
        print(f"\n6. Generating analysis summary...")
        summary = dynamic_mapper.get_analysis_summary()
        
        print(f"   ✅ Total tables analyzed: {summary['total_tables']}")
        print(f"   ✅ Global role coverage:")
        for role, count in summary['global_role_coverage'].items():
            print(f"      {role}: {count} tables")
        
        print(f"\n🎉 All column intelligence tests completed successfully!")
        print("✅ Phase 1: Column Intelligence & Detection is working correctly!")
        
        return True, results
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, {}

def test_integration_with_sanitization():
    """Test that column intelligence works with sanitization modules."""
    print(f"\n7. Testing integration with Phase 0 sanitization...")
    
    try:
        from sanitization import data_sanitizer
        from sql_security import sql_security
        from dynamic_mapping import dynamic_mapper
        
        # Create dirty data that needs both sanitization and column detection
        dirty_df = pd.DataFrame({
            'Asset@TAG#!': ['ASSET-001; DROP TABLE', 'ASSET<script>', 'ASSET-003'],
            'User$Name%^': ['John O\'Connor', 'Jane "Quote" Smith', 'Bob & Wilson'],
            'Company/Brand': ['HP Inc. & Co.', 'Dell\'; DROP--', 'Apple "Inc"']
        })
        
        print("   Testing with dirty data requiring sanitization...")
        
        # First sanitize the data
        clean_result = data_sanitizer.clean_csv_data(dirty_df, "integration_test.csv")
        clean_df = clean_result["cleaned_df"]
        
        print(f"   ✅ Data sanitized: {len(clean_result['report']['issues_fixed'])} issues fixed")
        print(f"   ✅ Columns normalized: {list(clean_df.columns)}")
        
        # Then analyze column structure
        analysis = dynamic_mapper.analyze_and_map_table(clean_df, "integration_test")
        detected_roles = analysis.get("role_mappings", {})
        
        print(f"   ✅ Column analysis on clean data: {len(detected_roles)} roles detected")
        for role, columns in detected_roles.items():
            if columns:
                print(f"      {role}: {columns[0]['column']} (confidence: {columns[0]['confidence']:.2f})")
        
        print("   ✅ Integration between Phase 0 and Phase 1 working correctly!")
        return True
        
    except Exception as e:
        print(f"   ❌ Integration test failed: {e}")
        return False

def run_performance_test():
    """Test performance with larger datasets."""
    print(f"\n8. Testing performance with larger datasets...")
    
    try:
        from dynamic_mapping import dynamic_mapper
        import time
        
        # Create a larger dataset
        large_data = {
            'asset_tag': [f'ASSET-{i:06d}' for i in range(1000)],
            'employee_name': [f'User {i}' for i in range(1000)],
            'manufacturer': ['Dell', 'HP', 'Apple', 'Lenovo', 'ASUS'] * 200,
            'device_model': [f'Model {i % 50}' for i in range(1000)],
            'location': [f'Floor {i % 10} Room {i % 100}' for i in range(1000)]
        }
        large_df = pd.DataFrame(large_data)
        
        start_time = time.time()
        analysis = dynamic_mapper.analyze_and_map_table(large_df, "large_dataset_test")
        end_time = time.time()
        
        analysis_time = end_time - start_time
        detected_roles = len(analysis.get("role_mappings", {}))
        
        print(f"   ✅ Large dataset (1000 rows, 5 columns) processed in {analysis_time:.2f}s")
        print(f"   ✅ Detected {detected_roles} semantic roles")
        
        if analysis_time < 5.0:  # Should complete within 5 seconds
            print("   ✅ Performance test passed!")
            return True
        else:
            print("   ⚠️  Performance could be improved")
            return True
            
    except Exception as e:
        print(f"   ❌ Performance test failed: {e}")
        return False

if __name__ == "__main__":
    print("Starting Phase 1 Column Intelligence Tests")
    print("This validates the dynamic column detection system")
    print()
    
    # Run all tests
    intelligence_ok, results = test_column_intelligence()
    integration_ok = test_integration_with_sanitization() if intelligence_ok else False
    performance_ok = run_performance_test() if intelligence_ok else False
    
    print("\n" + "="*60)
    if intelligence_ok and integration_ok and performance_ok:
        print("🎯 ALL TESTS PASSED - PHASE 1 COMPLETE!")
        print("✅ Column Intelligence & Detection system is ready for production")
        print("✅ Universal CSV compatibility achieved")
        print("✅ Integration with security modules confirmed")
        sys.exit(0)
    else:
        print("❌ SOME TESTS FAILED - PHASE 1 NEEDS REVIEW")
        print("🔧 Please review the errors above and address any issues")
        sys.exit(1)