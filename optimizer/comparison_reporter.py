"""
Generate comparison reports between baseline and GA results.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class ComparisonReporter:
    """Generate formatted comparison reports."""
    
    def __init__(self, output_dir: str = "results"):
        """Initialize reporter."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def generate_text_report(self, baseline: Dict, ga_results: Dict,
                            comparison: Optional[Dict] = None,
                            output_file: str = "comparison_report.txt") -> str:
        """Generate detailed text comparison report."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report_lines = [
            "=" * 80,
            "BLOOD SUPPLY ROUTE OPTIMIZATION - BASELINE vs GA COMPARISON",
            "=" * 80,
            f"Generated: {timestamp}\n",
            
            "HISTORICAL BASELINE METRICS",
            "-" * 80,
            f"Total Trips (historical): {baseline.get('num_trips', 0)}",
            f"Average Distance per Trip: {baseline.get('avg_distance_km', 0):.2f} km",
            f"Total Historical Distance: {baseline.get('total_distance_km', 0):.2f} km",
            f"Min/Max Distance: {baseline.get('min_distance_km', 0):.2f} - "
            f"{baseline.get('max_distance_km', 0):.2f} km",
        ]
        
        if baseline.get('avg_duration_hours'):
            report_lines.append(
                f"Average Duration per Trip: "
                f"{baseline.get('avg_duration_hours', 0):.2f} hours"
            )
        if comparison is not None:
            report_lines.extend([
                f"\nOn-Time Delivery Rate: {baseline.get('on_time_percentage', 0):.1f}%",
                f"  - On-time trips: {baseline.get('on_time_count', 0)}",
                f"  - Late trips: {baseline.get('late_count', 0)}",
                f"Average Lateness (late trips): {baseline.get('avg_lateness_minutes', 0):.1f} minutes",
            
                f"\nCost Analysis:",
                f"  - Cost per km: IDR {baseline.get('cost_per_km_idr', 0):,.0f}",
                f"  - Average cost per trip: IDR {baseline.get('avg_cost_per_trip_idr', 0):,.0f}",
                f"  - Total historical cost: IDR {baseline.get('total_cost_idr', 0):,.0f}",
                
                "\n\nGENETIC ALGORITHM RESULTS",
                "-" * 80,
                f"Number of Routes: {ga_results.get('num_routes', 0)}",
                f"Makespan (longest route): {ga_results.get('makespan_hours', 0):.2f} hours",
                f"Total Time (all routes): {ga_results.get('total_time_hours', 0):.2f} hours",
                f"Total Distance: {ga_results.get('total_distance_km', 0):.2f} km",
                f"Total Cost: IDR {ga_results.get('total_cost_idr', 0):,.0f}",
                
                "\n\nIMPROVEMENT METRICS",
                "-" * 80,
                f"Distance Reduction:",
                f"  - Absolute: {comparison.get('distance_reduction_km', 0):.2f} km",
                f"  - Percentage: {comparison.get('distance_reduction_pct', 0):.1f}%",
                
                f"Cost Reduction:",
                f"  - Absolute: IDR {comparison.get('cost_reduction_idr', 0):,.0f}",
                f"  - Percentage: {comparison.get('cost_reduction_pct', 0):.1f}%",
                
                "\n\nKEY INSIGHTS",
                "-" * 80,
                f"1. Historical on-time rate: {baseline.get('on_time_percentage', 0):.1f}%",
                f"2. GA produces {ga_results.get('num_routes', 0)} optimized routes",
                f"3. Expected distance reduction: {comparison.get('distance_reduction_pct', 0):.1f}%",
                f"4. Estimated cost savings: IDR {comparison.get('cost_reduction_idr', 0):,.0f}",
                
                "\n\nRECOMMENDATIONS",
                "-" * 80,
                "1. Implement optimized GA routes to reduce delivery distance",
                "2. Monitor actual delivery times to validate GA predictions",
                "3. Use baseline lateness as improvement target",
                "4. Cost savings can fund vehicle maintenance/fuel reserves",
                "5. Consider Deep Learning model to predict delays proactively",
                "",
                "=" * 80,
            ])
        else:
            report_lines.extend([
                f"\nOn-Time Delivery Rate: {baseline.get('on_time_percentage', 0):.1f}%",
                f"  - On-time trips: {baseline.get('on_time_count', 0)}",
                f"  - Late trips: {baseline.get('late_count', 0)}",
                f"Average Lateness (late trips): {baseline.get('avg_lateness_minutes', 0):.1f} minutes",
            
                f"\nCost Analysis:",
                f"  - Cost per km: IDR {baseline.get('cost_per_km_idr', 0):,.0f}",
                f"  - Average cost per trip: IDR {baseline.get('avg_cost_per_trip_idr', 0):,.0f}",
                f"  - Total historical cost: IDR {baseline.get('total_cost_idr', 0):,.0f}",
                
                "\n\nGENETIC ALGORITHM RESULTS",
                "-" * 80,
                f"Number of Routes: {ga_results.get('num_routes', 0)}",
                f"Makespan (longest route): {ga_results.get('makespan_hours', 0):.2f} hours",
                f"Total Time (all routes): {ga_results.get('total_time_hours', 0):.2f} hours",
                f"Total Distance: {ga_results.get('total_distance_km', 0):.2f} km",
                f"Total Cost: IDR {ga_results.get('total_cost_idr', 0):,.0f}",
                "\n\nNo comparison metrics available (baseline data missing)",
                "-" * 80,
            ])
        
        report_text = "\n".join(report_lines)
        
        # Save to file
        output_path = self.output_dir / output_file
        with open(output_path, 'w') as f:
            f.write(report_text)
        
        logger.info(f"Text report saved: {output_path}")
        return report_text
    
    def generate_json_report(self, baseline: Dict, ga_results: Dict,
                            comparison: Optional[Dict] = None,
                            output_file: str = "comparison.json") -> str:
        """Generate JSON comparison report."""
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'baseline': baseline,
            'ga_results': ga_results,
            'comparison': comparison,
        }
        
        output_path = self.output_dir / output_file
        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        logger.info(f"JSON report saved: {output_path}")
        return str(output_path)
    
    def generate_summary_table(self, baseline: Dict, ga_results: Dict,
                              comparison: Optional[Dict]) -> str:
        """Generate a summary comparison table."""
        comparison = comparison or {}
        lines = [
            "\nCOMPARISON TABLE",
            "-" * 100,
            f"{'Metric':<35} {'Baseline':<25} {'GA Result':<25} {'Improvement':<15}",
            "-" * 100,
        ]
        
        # Distance
        lines.append(
            f"{'Total Distance (km)':<35} "
            f"{baseline.get('total_distance_km', 0):>20.2f} "
            f"{ga_results.get('total_distance_km', 0):>20.2f} "
            f"{comparison.get('distance_reduction_pct', 0):>10.1f}%"
        )
        
        # Cost
        lines.append(
            f"{'Total Cost (IDR)':<35} "
            f"IDR {baseline.get('total_cost_idr', 0):>15,.0f} "
            f"IDR {ga_results.get('total_cost_idr', 0):>14,.0f} "
            f"{comparison.get('cost_reduction_pct', 0):>10.1f}%"
        )
        
        # Time
        lines.append(
            f"{'Total Time (hours)':<35} "
            f"{baseline.get('avg_duration_hours', 0) or 'N/A':>20} "
            f"{ga_results.get('total_time_hours', 0):>20.2f}"
        )
        
        # On-time rate
        lines.append(
            f"{'On-Time Delivery Rate':<35} "
            f"{baseline.get('on_time_percentage', 0):>20.1f}% "
            f"(Target)"
        )
        
        lines.extend([
            "-" * 100,
            ""
        ])
        
        return "\n".join(lines)
    
    def print_report(self, baseline: Dict, ga_results: Dict,
                    comparison: Optional[Dict]):
        """Print formatted report to console."""
        comparison = comparison or {}
        print("\n" + "=" * 80)
        print("GENETIC ALGORITHM OPTIMIZATION - BASELINE COMPARISON")
        print("=" * 80)
        
        print("\nBASELINE SUMMARY (Historical Data):")
        print("-" * 80)
        print(f"  Total trips analyzed: {baseline.get('num_trips', 0)}")
        print(f"  Average distance: {baseline.get('avg_distance_km', 0):.2f} km")
        print(f"  Total distance: {baseline.get('total_distance_km', 0):.2f} km")
        print(f"  On-time rate: {baseline.get('on_time_percentage', 0):.1f}%")
        print(f"  Total cost: IDR {baseline.get('total_cost_idr', 0):,.0f}")
        
        print("\nGA OPTIMIZATION RESULTS:")
        print("-" * 80)
        print(f"  Number of routes: {ga_results.get('num_routes', 0)}")
        print(f"  Makespan: {ga_results.get('makespan_hours', 0):.2f} hours")
        print(f"  Total distance: {ga_results.get('total_distance_km', 0):.2f} km")
        print(f"  Total cost: IDR {ga_results.get('total_cost_idr', 0):,.0f}")
        
        print("\nIMPROVEMENTS:")
        print("-" * 80)
        improvements = comparison.get('improvements', {})
        print(f"  Distance reduction: {improvements.get('distance_reduction_km', 0):.2f} km "
              f"({improvements.get('distance_reduction_pct', 0):.1f}%)")
        print(f"  Cost savings: IDR {improvements.get('cost_reduction_idr', 0):,.0f} "
              f"({improvements.get('cost_reduction_pct', 0):.1f}%)")
        
        print("\n" + "=" * 80 + "\n")
