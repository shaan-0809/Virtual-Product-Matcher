#!/usr/bin/env python3
"""
Performance monitoring script for the Visual Product Matcher application.
Tracks model loading times, image processing performance, and cache efficiency.
"""

import time
import json
import os
import psutil
import threading
from datetime import datetime
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import pandas as pd

class PerformanceMonitor:
    """Monitors and tracks application performance metrics"""
    
    def __init__(self, log_file='performance_log.json'):
        self.log_file = log_file
        self.metrics = {
            'model_load_times': [],
            'image_processing_times': [],
            'url_fetch_times': [],
            'search_times': [],
            'cache_hits': 0,
            'cache_misses': 0,
            'memory_usage': [],
            'cpu_usage': []
        }
        self.start_time = time.time()
        self.load_existing_logs()
    
    def load_existing_logs(self):
        """Load existing performance logs"""
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, 'r') as f:
                    existing_data = json.load(f)
                    for key in self.metrics:
                        if key in existing_data:
                            self.metrics[key] = existing_data[key]
            except Exception as e:
                print(f"Warning: Could not load existing logs: {e}")
    
    def log_model_load_time(self, load_time: float, model_type: str):
        """Log model loading time"""
        self.metrics['model_load_times'].append({
            'timestamp': datetime.now().isoformat(),
            'load_time': load_time,
            'model_type': model_type
        })
        self.save_logs()
    
    def log_image_processing_time(self, processing_time: float, image_size: tuple):
        """Log image processing time"""
        self.metrics['image_processing_times'].append({
            'timestamp': datetime.now().isoformat(),
            'processing_time': processing_time,
            'image_size': image_size
        })
        self.save_logs()
    
    def log_url_fetch_time(self, fetch_time: float, url: str, success: bool):
        """Log URL fetch time"""
        self.metrics['url_fetch_times'].append({
            'timestamp': datetime.now().isoformat(),
            'fetch_time': fetch_time,
            'url': url,
            'success': success
        })
        self.save_logs()
    
    def log_search_time(self, search_time: float, results_count: int):
        """Log search time"""
        self.metrics['search_times'].append({
            'timestamp': datetime.now().isoformat(),
            'search_time': search_time,
            'results_count': results_count
        })
        self.save_logs()
    
    def log_cache_hit(self):
        """Log cache hit"""
        self.metrics['cache_hits'] += 1
        self.save_logs()
    
    def log_cache_miss(self):
        """Log cache miss"""
        self.metrics['cache_misses'] += 1
        self.save_logs()
    
    def log_system_metrics(self):
        """Log system resource usage"""
        self.metrics['memory_usage'].append({
            'timestamp': datetime.now().isoformat(),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_available': psutil.virtual_memory().available / (1024**3)  # GB
        })
        self.metrics['cpu_usage'].append({
            'timestamp': datetime.now().isoformat(),
            'cpu_percent': psutil.cpu_percent(interval=1)
        })
        self.save_logs()
    
    def save_logs(self):
        """Save metrics to log file"""
        try:
            with open(self.log_file, 'w') as f:
                json.dump(self.metrics, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save logs: {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics"""
        summary = {
            'total_runtime': time.time() - self.start_time,
            'model_loads': len(self.metrics['model_load_times']),
            'image_processing_operations': len(self.metrics['image_processing_times']),
            'url_fetches': len(self.metrics['url_fetch_times']),
            'searches': len(self.metrics['search_times']),
            'cache_hit_rate': 0,
            'average_model_load_time': 0,
            'average_image_processing_time': 0,
            'average_url_fetch_time': 0,
            'average_search_time': 0
        }
        
        # Calculate averages
        if self.metrics['model_load_times']:
            summary['average_model_load_time'] = sum(
                m['load_time'] for m in self.metrics['model_load_times']
            ) / len(self.metrics['model_load_times'])
        
        if self.metrics['image_processing_times']:
            summary['average_image_processing_time'] = sum(
                m['processing_time'] for m in self.metrics['image_processing_times']
            ) / len(self.metrics['image_processing_times'])
        
        if self.metrics['url_fetch_times']:
            summary['average_url_fetch_time'] = sum(
                m['fetch_time'] for m in self.metrics['url_fetch_times']
            ) / len(self.metrics['url_fetch_times'])
        
        if self.metrics['search_times']:
            summary['average_search_time'] = sum(
                m['search_time'] for m in self.metrics['search_times']
            ) / len(self.metrics['search_times'])
        
        # Calculate cache hit rate
        total_cache_ops = self.metrics['cache_hits'] + self.metrics['cache_misses']
        if total_cache_ops > 0:
            summary['cache_hit_rate'] = self.metrics['cache_hits'] / total_cache_ops
        
        return summary
    
    def generate_performance_report(self, output_file='performance_report.html'):
        """Generate HTML performance report"""
        summary = self.get_performance_summary()
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Performance Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .metric {{ background: #f5f5f5; padding: 10px; margin: 10px 0; border-radius: 5px; }}
                .good {{ color: green; }}
                .warning {{ color: orange; }}
                .poor {{ color: red; }}
            </style>
        </head>
        <body>
            <h1>Performance Report</h1>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="metric">
                <h3>Overall Statistics</h3>
                <p>Total Runtime: {summary['total_runtime']:.2f} seconds</p>
                <p>Model Loads: {summary['model_loads']}</p>
                <p>Image Processing Operations: {summary['image_processing_operations']}</p>
                <p>URL Fetches: {summary['url_fetches']}</p>
                <p>Searches: {summary['searches']}</p>
            </div>
            
            <div class="metric">
                <h3>Performance Metrics</h3>
                <p>Average Model Load Time: {summary['average_model_load_time']:.3f}s</p>
                <p>Average Image Processing Time: {summary['average_image_processing_time']:.3f}s</p>
                <p>Average URL Fetch Time: {summary['average_url_fetch_time']:.3f}s</p>
                <p>Average Search Time: {summary['average_search_time']:.3f}s</p>
                <p>Cache Hit Rate: {summary['cache_hit_rate']:.2%}</p>
            </div>
            
            <div class="metric">
                <h3>Recommendations</h3>
                {self._generate_recommendations(summary)}
            </div>
        </body>
        </html>
        """
        
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        print(f"Performance report generated: {output_file}")
    
    def _generate_recommendations(self, summary: Dict[str, Any]) -> str:
        """Generate performance recommendations"""
        recommendations = []
        
        if summary['average_model_load_time'] > 5.0:
            recommendations.append("Model load time is high. Consider using model caching.")
        
        if summary['average_url_fetch_time'] > 3.0:
            recommendations.append("URL fetch time is high. Consider implementing better caching.")
        
        if summary['cache_hit_rate'] < 0.5:
            recommendations.append("Cache hit rate is low. Consider expanding cache size.")
        
        if summary['average_search_time'] > 2.0:
            recommendations.append("Search time is high. Consider optimizing similarity search algorithm.")
        
        if not recommendations:
            recommendations.append("Performance looks good! No immediate optimizations needed.")
        
        return "<ul>" + "".join(f"<li>{rec}</li>" for rec in recommendations) + "</ul>"

def start_system_monitoring(monitor: PerformanceMonitor, interval: int = 30):
    """Start background system monitoring"""
    def monitor_loop():
        while True:
            monitor.log_system_metrics()
            time.sleep(interval)
    
    thread = threading.Thread(target=monitor_loop, daemon=True)
    thread.start()
    return thread

if __name__ == "__main__":
    # Example usage
    monitor = PerformanceMonitor()
    
    # Start system monitoring
    monitoring_thread = start_system_monitoring(monitor)
    
    # Example metrics logging
    monitor.log_model_load_time(2.5, "EfficientNetB0")
    monitor.log_image_processing_time(0.3, (224, 224))
    monitor.log_url_fetch_time(1.2, "https://example.com/image.jpg", True)
    monitor.log_search_time(0.8, 10)
    monitor.log_cache_hit()
    monitor.log_cache_miss()
    
    # Generate report
    monitor.generate_performance_report()
    
    print("Performance monitoring example completed.")
