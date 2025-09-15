#!/usr/bin/env python3
"""
Color Detection and Analysis Module
This module provides comprehensive color detection and analysis for images.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from collections import Counter
import webcolors
import argparse
import os
import json
from PIL import Image
import colorsys

from color_utils import ColorUtils

class ColorDetector:
    """Main color detection and analysis class"""
    
    def __init__(self):
        self.color_utils = ColorUtils()
        self.results = {}
    
    def load_image(self, image_path):
        """
        Load image from file path
        
        Args:
            image_path: Path to image file
        
        Returns:
            Loaded image in RGB format
        """
        try:
            # Load with OpenCV (BGR format)
            image_bgr = cv2.imread(image_path)
            if image_bgr is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            
            print(f"[INFO] Loaded image: {image_path}")
            print(f"[INFO] Image shape: {image_rgb.shape}")
            
            return image_rgb
            
        except Exception as e:
            print(f"[ERROR] Failed to load image: {e}")
            return None
    
    def extract_dominant_colors(self, image, num_colors=5, resize_factor=0.5):
        """
        Extract dominant colors using K-means clustering
        
        Args:
            image: Input image (RGB format)
            num_colors: Number of dominant colors to extract
            resize_factor: Factor to resize image for faster processing
        
        Returns:
            List of dominant colors with percentages
        """
        try:
            # Resize image for faster processing
            if resize_factor < 1.0:
                height, width = image.shape[:2]
                new_height = int(height * resize_factor)
                new_width = int(width * resize_factor)
                image = cv2.resize(image, (new_width, new_height))
            
            # Reshape image to be a list of pixels
            pixels = image.reshape(-1, 3)
            
            # Apply K-means clustering
            kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init=10)
            kmeans.fit(pixels)
            
            # Get cluster centers (dominant colors)
            colors = kmeans.cluster_centers_.astype(int)
            
            # Get labels for each pixel
            labels = kmeans.labels_
            
            # Count occurrences of each cluster
            label_counts = Counter(labels)
            total_pixels = len(labels)
            
            # Create results with percentages
            dominant_colors = []
            for i, color in enumerate(colors):
                percentage = (label_counts[i] / total_pixels) * 100
                color_info = {
                    'rgb': tuple(color),
                    'hex': self.color_utils.rgb_to_hex(color),
                    'percentage': round(percentage, 2),
                    'color_name': self.color_utils.get_color_name(color)
                }
                dominant_colors.append(color_info)
            
            # Sort by percentage (descending)
            dominant_colors.sort(key=lambda x: x['percentage'], reverse=True)
            
            print(f"[INFO] Extracted {len(dominant_colors)} dominant colors")
            
            return dominant_colors
            
        except Exception as e:
            print(f"[ERROR] Failed to extract dominant colors: {e}")
            return []
    
    def analyze_color_distribution(self, image):
        """
        Analyze color distribution in the image
        
        Args:
            image: Input image (RGB format)
        
        Returns:
            Dictionary with color distribution statistics
        """
        try:
            # Convert to different color spaces
            hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            
            # Calculate histograms
            hist_r = cv2.calcHist([image], [0], None, [256], [0, 256])
            hist_g = cv2.calcHist([image], [1], None, [256], [0, 256])
            hist_b = cv2.calcHist([image], [2], None, [256], [0, 256])
            
            # Calculate statistics
            pixels = image.reshape(-1, 3)
            
            stats = {
                'total_pixels': len(pixels),
                'unique_colors': len(np.unique(pixels.view(np.dtype((np.void, pixels.dtype.itemsize*pixels.shape[1])))))
            }
            
            # RGB statistics
            stats['rgb_stats'] = {
                'red': {
                    'mean': float(np.mean(pixels[:, 0])),
                    'std': float(np.std(pixels[:, 0])),
                    'min': int(np.min(pixels[:, 0])),
                    'max': int(np.max(pixels[:, 0]))
                },
                'green': {
                    'mean': float(np.mean(pixels[:, 1])),
                    'std': float(np.std(pixels[:, 1])),
                    'min': int(np.min(pixels[:, 1])),
                    'max': int(np.max(pixels[:, 1]))
                },
                'blue': {
                    'mean': float(np.mean(pixels[:, 2])),
                    'std': float(np.std(pixels[:, 2])),
                    'min': int(np.min(pixels[:, 2])),
                    'max': int(np.max(pixels[:, 2]))
                }
            }
            
            # HSV statistics
            hsv_pixels = hsv_image.reshape(-1, 3)
            stats['hsv_stats'] = {
                'hue': {
                    'mean': float(np.mean(hsv_pixels[:, 0])),
                    'std': float(np.std(hsv_pixels[:, 0]))
                },
                'saturation': {
                    'mean': float(np.mean(hsv_pixels[:, 1])),
                    'std': float(np.std(hsv_pixels[:, 1]))
                },
                'value': {
                    'mean': float(np.mean(hsv_pixels[:, 2])),
                    'std': float(np.std(hsv_pixels[:, 2]))
                }
            }
            
            # Overall brightness
            brightness = np.mean(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))
            stats['brightness'] = float(brightness)
            
            # Color temperature estimation
            stats['color_temperature'] = self.estimate_color_temperature(image)
            
            return stats
            
        except Exception as e:
            print(f"[ERROR] Failed to analyze color distribution: {e}")
            return {}
    
    def estimate_color_temperature(self, image):
        """
        Estimate color temperature of the image
        
        Args:
            image: Input image (RGB format)
        
        Returns:
            Estimated color temperature in Kelvin
        """
        try:
            # Calculate average RGB values
            avg_r = np.mean(image[:, :, 0])
            avg_g = np.mean(image[:, :, 1])
            avg_b = np.mean(image[:, :, 2])
            
            # Simple color temperature estimation
            # This is a simplified method - more accurate methods exist
            if avg_b > avg_r:
                # Cool temperature (bluish)
                temp = 6500 + (avg_b - avg_r) * 10
            else:
                # Warm temperature (reddish)
                temp = 6500 - (avg_r - avg_b) * 10
            
            return max(2000, min(10000, int(temp)))
            
        except Exception as e:
            print(f"[ERROR] Failed to estimate color temperature: {e}")
            return 6500  # Default daylight temperature
    
    def create_color_palette(self, dominant_colors, palette_size=(400, 100)):
        """
        Create a visual color palette from dominant colors
        
        Args:
            dominant_colors: List of dominant color information
            palette_size: Size of the palette image (width, height)
        
        Returns:
            PIL Image of the color palette
        """
        try:
            width, height = palette_size
            palette_image = Image.new('RGB', (width, height))
            
            # Calculate width for each color
            num_colors = len(dominant_colors)
            if num_colors == 0:
                return palette_image
            
            color_width = width // num_colors
            
            # Draw color blocks
            for i, color_info in enumerate(dominant_colors):
                color = color_info['rgb']
                x_start = i * color_width
                x_end = x_start + color_width
                
                # Fill the color block
                for x in range(x_start, min(x_end, width)):
                    for y in range(height):
                        palette_image.putpixel((x, y), color)
            
            return palette_image
            
        except Exception as e:
            print(f"[ERROR] Failed to create color palette: {e}")
            return Image.new('RGB', palette_size, (255, 255, 255))
    
    def analyze_image(self, image_path, num_colors=5, save_results=True):
        """
        Perform complete color analysis on an image
        
        Args:
            image_path: Path to the image file
            num_colors: Number of dominant colors to extract
            save_results: Whether to save results to file
        
        Returns:
            Complete analysis results dictionary
        """
        try:
            # Load image
            image = self.load_image(image_path)
            if image is None:
                return None
            
            print(f"[INFO] Analyzing image: {os.path.basename(image_path)}")
            
            # Extract dominant colors
            dominant_colors = self.extract_dominant_colors(image, num_colors)
            
            # Analyze color distribution
            distribution_stats = self.analyze_color_distribution(image)
            
            # Create color palette
            palette_image = self.create_color_palette(dominant_colors)
            
            # Compile results
            results = {
                'image_path': image_path,
                'image_name': os.path.basename(image_path),
                'image_size': image.shape,
                'dominant_colors': dominant_colors,
                'distribution_stats': distribution_stats,
                'analysis_timestamp': self.color_utils.get_timestamp()
            }
            
            # Save results if requested
            if save_results:
                self.save_results(results, image_path)
                
                # Save palette image
                palette_path = self.get_output_path(image_path, '_palette.png')
                palette_image.save(palette_path)
                print(f"[INFO] Color palette saved: {palette_path}")
            
            self.results = results
            return results
            
        except Exception as e:
            print(f"[ERROR] Failed to analyze image: {e}")
            return None
    
    def save_results(self, results, image_path):
        """
        Save analysis results to JSON file
        
        Args:
            results: Analysis results dictionary
            image_path: Original image path
        """
        try:
            output_path = self.get_output_path(image_path, '_analysis.json')
            
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            print(f"[INFO] Results saved: {output_path}")
            
        except Exception as e:
            print(f"[ERROR] Failed to save results: {e}")
    
    def get_output_path(self, image_path, suffix):
        """
        Generate output file path
        
        Args:
            image_path: Original image path
            suffix: Suffix to add to filename
        
        Returns:
            Output file path
        """
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_dir = "results"
        os.makedirs(output_dir, exist_ok=True)
        return os.path.join(output_dir, f"{base_name}{suffix}")
    
    def visualize_results(self, results, show_plot=True, save_plot=True):
        """
        Create visualization of color analysis results
        
        Args:
            results: Analysis results dictionary
            show_plot: Whether to display the plot
            save_plot: Whether to save the plot
        """
        try:
            if not results or 'dominant_colors' not in results:
                print("[ERROR] No results to visualize")
                return
            
            dominant_colors = results['dominant_colors']
            
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f"Color Analysis: {results['image_name']}", fontsize=16)
            
            # 1. Dominant colors bar chart
            colors = [color['rgb'] for color in dominant_colors]
            percentages = [color['percentage'] for color in dominant_colors]
            color_names = [color['color_name'] for color in dominant_colors]
            
            # Normalize colors for matplotlib
            colors_normalized = [(r/255, g/255, b/255) for r, g, b in colors]
            
            axes[0, 0].bar(range(len(colors)), percentages, color=colors_normalized)
            axes[0, 0].set_title('Dominant Colors')
            axes[0, 0].set_xlabel('Color Index')
            axes[0, 0].set_ylabel('Percentage (%)')
            axes[0, 0].set_xticks(range(len(colors)))
            axes[0, 0].set_xticklabels([f"C{i+1}" for i in range(len(colors))])
            
            # 2. Color palette
            palette_height = 1
            for i, color in enumerate(colors_normalized):
                axes[0, 1].barh(0, 1, left=i, height=palette_height, color=color)
            
            axes[0, 1].set_title('Color Palette')
            axes[0, 1].set_xlim(0, len(colors))
            axes[0, 1].set_ylim(-0.5, 0.5)
            axes[0, 1].set_yticks([])
            axes[0, 1].set_xticks(range(len(colors)))
            axes[0, 1].set_xticklabels([f"C{i+1}" for i in range(len(colors))])
            
            # 3. RGB distribution
            if 'distribution_stats' in results and 'rgb_stats' in results['distribution_stats']:
                rgb_stats = results['distribution_stats']['rgb_stats']
                channels = ['red', 'green', 'blue']
                means = [rgb_stats[ch]['mean'] for ch in channels]
                
                axes[1, 0].bar(channels, means, color=['red', 'green', 'blue'], alpha=0.7)
                axes[1, 0].set_title('Average RGB Values')
                axes[1, 0].set_ylabel('Average Value')
                axes[1, 0].set_ylim(0, 255)
            
            # 4. Color information table
            axes[1, 1].axis('tight')
            axes[1, 1].axis('off')
            
            table_data = []
            for i, color in enumerate(dominant_colors):
                table_data.append([
                    f"C{i+1}",
                    color['hex'],
                    f"{color['percentage']:.1f}%",
                    color['color_name'][:15]  # Truncate long names
                ])
            
            table = axes[1, 1].table(cellText=table_data,
                                   colLabels=['Color', 'HEX', 'Percentage', 'Name'],
                                   cellLoc='center',
                                   loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.2, 1.5)
            axes[1, 1].set_title('Color Details')
            
            plt.tight_layout()
            
            if save_plot:
                plot_path = self.get_output_path(results['image_path'], '_visualization.png')
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                print(f"[INFO] Visualization saved: {plot_path}")
            
            if show_plot:
                plt.show()
            else:
                plt.close()
                
        except Exception as e:
            print(f"[ERROR] Failed to create visualization: {e}")

def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description="Color Detection and Analysis")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--dominant", type=int, default=5, help="Number of dominant colors to extract")
    parser.add_argument("--no-save", action="store_true", help="Don't save results to files")
    parser.add_argument("--no-viz", action="store_true", help="Don't create visualizations")
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"[ERROR] Image file not found: {args.image}")
        return
    
    # Initialize detector
    detector = ColorDetector()
    
    # Analyze image
    results = detector.analyze_image(
        args.image, 
        num_colors=args.dominant,
        save_results=not args.no_save
    )
    
    if results:
        # Print summary
        print(f"\n[INFO] Analysis Summary:")
        print(f"Image: {results['image_name']}")
        print(f"Size: {results['image_size']}")
        print(f"Dominant Colors:")
        
        for i, color in enumerate(results['dominant_colors']):
            print(f"  {i+1}. {color['color_name']} ({color['hex']}) - {color['percentage']:.1f}%")
        
        if 'distribution_stats' in results:
            stats = results['distribution_stats']
            print(f"Brightness: {stats.get('brightness', 0):.1f}")
            print(f"Color Temperature: {stats.get('color_temperature', 0)}K")
            print(f"Unique Colors: {stats.get('unique_colors', 0)}")
        
        # Create visualization
        if not args.no_viz:
            detector.visualize_results(results)

if __name__ == "__main__":
    main()

