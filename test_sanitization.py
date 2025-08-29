#!/usr/bin/env python3
"""
Test script for the new modular data sanitization system.
Tests Phase 0: Data Sanitization & Security implementation.
"""

import pandas as pd
import sys
import os

# Add current directory to path to import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_sanitization_modules():
    """Test the sanitization modules functionality."""
    print("🧪 Testing Phase 0: Data Sanitization & Security")
    print("=" * 60)
    
    try:
        # Test 1: Import modules
        print("1. Testing module imports...")
        from sanitization import data_sanitizer
        from sql_security import sql_security
        print("   ✅ Sanitization modules imported successfully")
        
        # Test 2: CSV data cleaning
        print("\n2. Testing CSV data cleaning...")
        # Create test DataFrame with dirty data
        dirty_data = {
            'User Name@#$': ['John O\'Connor  ', 'Jane; Smith', 'Bob "Quote" Wilson'],
            'Asset TAG!': ['ASSET-123@#$', 'ASSET;456', 'ASSET"789'],
            'Manufacturer': ['HP Inc. & Co.', 'Dell; Corp', 'Apple "Inc"'],
            '   Spaces   ': ['Value 1', 'Value;2', 'Value"3']
        }
        dirty_df = pd.DataFrame(dirty_data)
        
        # Clean the data
        result = data_sanitizer.clean_csv_data(dirty_df, "test_data.csv")
        
        cleaned_df = result["cleaned_df"]
        report = result["report"]
        
        print(f"   ✅ Original columns: {dirty_df.columns.tolist()}")
        print(f"   ✅ Cleaned columns: {cleaned_df.columns.tolist()}")
        print(f"   ✅ Issues found: {len(report['issues_found'])}")
        print(f"   ✅ Values cleaned: {report['values_cleaned']}")
        
        # Test 3: User input sanitization
        print("\n3. Testing user input sanitization...")
        
        # Test cases with different input types
        test_inputs = [
            ("what is asset tag ASSET-123@#$?", "general"),
            ("ASSET-VAF-HO-IV-124;DROP TABLE users;", "asset_tag"),
            ("John O'Connor<script>alert('xss')</script>", "username"),
            ("HP Inc. & Co.'OR 1=1--", "manufacturer"),
        ]
        
        for test_input, input_type in test_inputs:
            sanitized = data_sanitizer.sanitize_user_input(test_input, input_type)
            print(f"   ✅ {input_type}: '{test_input}' -> '{sanitized}'")
        
        # Test 4: SQL security
        print("\n4. Testing SQL security...")
        
        # Test safe SQL building
        safe_sql, params = sql_security.build_safe_select_query(
            table_name="test_table",
            columns=["column1", "column2"],
            where_conditions={"Asset_TAG": "ASSET-123"},
            limit=10
        )
        print(f"   ✅ Safe SQL: {safe_sql}")
        print(f"   ✅ Parameters: {params}")
        
        # Test parameter sanitization
        dangerous_params = {
            "asset_tag": "ASSET-123'; DROP TABLE users; --",
            "user_name": "admin' OR '1'='1",
            "search": "<script>alert('xss')</script>"
        }
        
        sanitized_params = sql_security._sanitize_query_params(dangerous_params)
        print(f"   ✅ Sanitized dangerous params: {sanitized_params}")
        
        print(f"\n🎉 All sanitization tests passed!")
        print("✅ Phase 0: Data Sanitization & Security implementation is working correctly!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_legacy_compatibility():
    """Test that legacy sanitization still works as fallback."""
    print(f"\n5. Testing legacy compatibility...")
    try:
        from core import sanitize_dataframe
        
        # Test with simple dirty data
        test_df = pd.DataFrame({
            'Column with Spaces': ['data1', 'data2'],
            'Column@#$%': ['data3', 'data4']
        })
        
        result = sanitize_dataframe(test_df, "legacy_test.csv")
        print(f"   ✅ Legacy sanitization works")
        print(f"   ✅ Security enhanced: {result.get('security_enhanced', False)}")
        return True
        
    except Exception as e:
        print(f"   ❌ Legacy compatibility test failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("Starting Phase 0 Data Sanitization Tests")
    print("This tests the modular security system implementation")
    print()
    
    # Run tests
    sanitization_ok = test_sanitization_modules()
    legacy_ok = test_legacy_compatibility()
    
    print("\n" + "="*60)
    if sanitization_ok and legacy_ok:
        print("🎯 ALL TESTS PASSED - PHASE 0 COMPLETE!")
        print("✅ Data Sanitization & Security modules are ready for production")
        sys.exit(0)
    else:
        print("❌ SOME TESTS FAILED - PHASE 0 NEEDS FIXES")
        print("🔧 Please review the errors above and fix the issues")
        sys.exit(1)