#!/usr/bin/env python3
"""
Test script to verify the fixes for the issues found in debug/issue.txt
"""

import sys
import requests
import time

def test_upload_and_indexing():
    """Test issue #1: MySQL indexing with NaN values"""
    print("🧪 Testing upload and indexing with NaN values...")
    
    # Upload the titanic data that has NaN values
    with open('debug/titanic-data.csv', 'rb') as f:
        files = {'file': ('titanic-data.csv', f, 'text/csv')}
        response = requests.post('http://localhost:8000/upload', files=files)
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Upload successful: {data['filename']} with {data['shape'][0]} rows")
        return data['dataset_id']
    else:
        print(f"❌ Upload failed: {response.text}")
        return None

def test_file_operations(dataset_id):
    """Test issue #2: File operations and redis_client"""
    print("🧪 Testing file operations...")
    
    # Test file listing
    response = requests.get('http://localhost:8000/files')
    if response.status_code == 200:
        files = response.json()['files']
        print(f"✅ File listing successful: {len(files)} files found")
        
        if files:
            filename = files[0]['filename']
            
            # Test file deletion (should fix redis_client error)
            print(f"🧪 Testing delete for {filename}...")
            response = requests.delete(f'http://localhost:8000/files/{filename}')
            if response.status_code == 200:
                print("✅ File deletion successful")
                return True
            else:
                print(f"❌ File deletion failed: {response.text}")
                return False
    else:
        print(f"❌ File listing failed: {response.text}")
        return False

def test_natural_language_queries():
    """Test issue #3 & #4: Natural language queries"""
    print("🧪 Testing natural language queries...")
    
    # First re-upload since we deleted the file
    dataset_id = test_upload_and_indexing()
    if not dataset_id:
        return False
    
    # Wait a moment for processing
    time.sleep(2)
    
    # Test Albert Wirz query
    print("🧪 Testing Albert Wirz query...")
    response = requests.post('http://localhost:8000/chat', 
                           json={'question': 'tell me about albert wirz'})
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Albert Wirz query result: {data.get('summary', 'No summary')}")
    else:
        print(f"❌ Albert Wirz query failed: {response.text}")
    
    # Test passenger count query  
    print("🧪 Testing passenger count query...")
    response = requests.post('http://localhost:8000/chat',
                           json={'question': 'how many total passengers?'})
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Passenger count result: {data.get('summary', 'No summary')}")
    else:
        print(f"❌ Passenger count query failed: {response.text}")

if __name__ == "__main__":
    print("🚀 Starting fix verification tests...")
    
    # Test 1: Upload with NaN handling
    dataset_id = test_upload_and_indexing()
    
    if dataset_id:
        # Test 2: File operations
        test_file_operations(dataset_id)
        
        # Test 3 & 4: Natural language queries
        test_natural_language_queries()
    
    print("🏁 Test completed!")