#!/usr/bin/env python3
"""
Script to explore and analyze the hotel PMS data JSON schema
"""

import json
from collections import defaultdict, Counter
from datetime import datetime
import sys

def analyze_json_schema(filename, sample_size=10000):
    """
    Analyze the structure and content of the hotel PMS JSON data
    """
    print(f"🏨 Analyzing Hotel PMS Data Schema")
    print(f"📄 File: {filename}")
    print("=" * 60)
    
    # Track schema information
    field_types = defaultdict(set)
    field_examples = defaultdict(list)
    record_count = 0
    
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
            
        print(f"📊 Total records: {len(data):,}")
        
        # Sample records for analysis
        sample_data = data[:min(sample_size, len(data))]
        print(f"🔍 Analyzing sample of {len(sample_data):,} records")
        print()
        
        # Analyze each record
        for record in sample_data:
            record_count += 1
            for field, value in record.items():
                field_types[field].add(type(value).__name__)
                if len(field_examples[field]) < 5:  # Keep first 5 examples
                    field_examples[field].append(value)
        
        # Display schema analysis
        print("📋 FIELD SCHEMA ANALYSIS")
        print("-" * 60)
        
        for field in sorted(field_types.keys()):
            types = ', '.join(sorted(field_types[field]))
            examples = field_examples[field][:3]  # Show first 3 examples
            
            print(f"🔸 {field}")
            print(f"   Type(s): {types}")
            print(f"   Examples: {examples}")
            print()
        
        # Business insights
        print("🎯 BUSINESS DATA INSIGHTS")
        print("-" * 60)
        
        # Hotel information
        hotels = set()
        room_types = Counter()
        meal_types = Counter()
        date_range = []
        prices = []
        availability_counts = []
        
        for record in sample_data:
            hotels.add((record.get('raw_hotel_id'), record.get('raw_hotel_name')))
            room_types[record.get('raw_room_name')] += 1
            meal_types[record.get('raw_meal')] += 1
            
            if record.get('raw_check_in_date'):
                date_range.append(record.get('raw_check_in_date'))
            
            if record.get('raw_price_amount'):
                prices.append(record.get('raw_price_amount'))
                
            if record.get('raw_availability') is not None:
                availability_counts.append(record.get('raw_availability'))
        
        print(f"🏨 Hotels in dataset: {len(hotels)}")
        for hotel_id, hotel_name in sorted(hotels):
            print(f"   • {hotel_name} (ID: {hotel_id})")
        
        print(f"\n🛏️  Room Types ({len(room_types)} types):")
        for room_type, count in room_types.most_common():
            percentage = (count / len(sample_data)) * 100
            print(f"   • {room_type}: {count:,} records ({percentage:.1f}%)")
        
        print(f"\n🍽️  Meal Options:")
        for meal, count in meal_types.most_common():
            percentage = (count / len(sample_data)) * 100
            print(f"   • {meal}: {count:,} records ({percentage:.1f}%)")
        
        if date_range:
            print(f"\n📅 Date Range:")
            print(f"   • From: {min(date_range)}")
            print(f"   • To: {max(date_range)}")
            print(f"   • Unique dates: {len(set(date_range))}")
        
        if prices:
            print(f"\n💰 Price Analysis (EUR):")
            print(f"   • Min: €{min(prices)}")
            print(f"   • Max: €{max(prices)}")
            print(f"   • Average: €{sum(prices)/len(prices):.2f}")
        
        if availability_counts:
            print(f"\n🏪 Availability Analysis:")
            print(f"   • Min availability: {min(availability_counts)}")
            print(f"   • Max availability: {max(availability_counts)}")
            print(f"   • Average: {sum(availability_counts)/len(availability_counts):.2f}")
            
            # Sold out analysis
            sold_out_count = sum(1 for record in sample_data if record.get('raw_room_is_sold_out'))
            print(f"   • Sold out records: {sold_out_count:,} ({(sold_out_count/len(sample_data))*100:.1f}%)")
        
        print("\n" + "=" * 60)
        print("✅ Schema analysis complete!")
        
    except Exception as e:
        print(f"❌ Error analyzing file: {e}")
        return False
    
    return True

if __name__ == "__main__":
    filename = "raw_hotel_pms_data.json"
    analyze_json_schema(filename)
