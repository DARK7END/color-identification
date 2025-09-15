#!/usr/bin/env python3
"""
Color Utilities Module
Provides utility functions for color conversion and manipulation.
"""

import webcolors
import colorsys
from datetime import datetime
import numpy as np
import json
import os

class ColorUtils:
    """Utility class for color operations"""
    
    def __init__(self):
        self.color_names_db = self.load_color_names()
    
    def load_color_names(self):
        """Load color names database"""
        # Extended color names dictionary
        color_names = {
            # Basic colors
            (255, 0, 0): "Red",
            (0, 255, 0): "Green", 
            (0, 0, 255): "Blue",
            (255, 255, 0): "Yellow",
            (255, 0, 255): "Magenta",
            (0, 255, 255): "Cyan",
            (255, 255, 255): "White",
            (0, 0, 0): "Black",
            (128, 128, 128): "Gray",
            
            # Extended colors
            (255, 165, 0): "Orange",
            (128, 0, 128): "Purple",
            (255, 192, 203): "Pink",
            (165, 42, 42): "Brown",
            (255, 215, 0): "Gold",
            (192, 192, 192): "Silver",
            (128, 0, 0): "Maroon",
            (0, 128, 0): "Dark Green",
            (0, 0, 128): "Navy",
            (128, 128, 0): "Olive",
            (0, 128, 128): "Teal",
            (255, 20, 147): "Deep Pink",
            (255, 69, 0): "Red Orange",
            (50, 205, 50): "Lime Green",
            (135, 206, 235): "Sky Blue",
            (255, 218, 185): "Peach",
            (221, 160, 221): "Plum",
            (255, 240, 245): "Lavender Blush",
            (240, 248, 255): "Alice Blue",
            (245, 245, 220): "Beige"
        }
        
        return color_names
    
    def rgb_to_hex(self, rgb):
        """
        Convert RGB to HEX format
        
        Args:
            rgb: RGB tuple (r, g, b)
        
        Returns:
            HEX color string
        """
        try:
            if isinstance(rgb, (list, tuple, np.ndarray)):
                r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
                return f"#{r:02x}{g:02x}{b:02x}"
            else:
                return "#000000"
        except Exception:
            return "#000000"
    
    def hex_to_rgb(self, hex_color):
        """
        Convert HEX to RGB format
        
        Args:
            hex_color: HEX color string
        
        Returns:
            RGB tuple (r, g, b)
        """
        try:
            hex_color = hex_color.lstrip('#')
            return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        except Exception:
            return (0, 0, 0)
    
    def rgb_to_hsv(self, rgb):
        """
        Convert RGB to HSV format
        
        Args:
            rgb: RGB tuple (r, g, b)
        
        Returns:
            HSV tuple (h, s, v)
        """
        try:
            r, g, b = rgb[0]/255.0, rgb[1]/255.0, rgb[2]/255.0
            h, s, v = colorsys.rgb_to_hsv(r, g, b)
            return (int(h*360), int(s*100), int(v*100))
        except Exception:
            return (0, 0, 0)
    
    def hsv_to_rgb(self, hsv):
        """
        Convert HSV to RGB format
        
        Args:
            hsv: HSV tuple (h, s, v)
        
        Returns:
            RGB tuple (r, g, b)
        """
        try:
            h, s, v = hsv[0]/360.0, hsv[1]/100.0, hsv[2]/100.0
            r, g, b = colorsys.hsv_to_rgb(h, s, v)
            return (int(r*255), int(g*255), int(b*255))
        except Exception:
            return (0, 0, 0)
    
    def rgb_to_cmyk(self, rgb):
        """
        Convert RGB to CMYK format
        
        Args:
            rgb: RGB tuple (r, g, b)
        
        Returns:
            CMYK tuple (c, m, y, k)
        """
        try:
            r, g, b = rgb[0]/255.0, rgb[1]/255.0, rgb[2]/255.0
            
            k = 1 - max(r, g, b)
            if k == 1:
                return (0, 0, 0, 100)
            
            c = (1 - r - k) / (1 - k)
            m = (1 - g - k) / (1 - k)
            y = (1 - b - k) / (1 - k)
            
            return (int(c*100), int(m*100), int(y*100), int(k*100))
        except Exception:
            return (0, 0, 0, 0)
    
    def get_color_name(self, rgb):
        """
        Get the closest color name for RGB values
        
        Args:
            rgb: RGB tuple (r, g, b)
        
        Returns:
            Color name string
        """
        try:
            # First try exact match
            rgb_tuple = tuple(int(c) for c in rgb)
            if rgb_tuple in self.color_names_db:
                return self.color_names_db[rgb_tuple]
            
            # Try webcolors library
            try:
                color_name = webcolors.rgb_to_name(rgb_tuple)
                return color_name.title()
            except ValueError:
                pass
            
            # Find closest color by distance
            min_distance = float('inf')
            closest_color = "Unknown"
            
            for known_rgb, name in self.color_names_db.items():
                distance = self.color_distance(rgb_tuple, known_rgb)
                if distance < min_distance:
                    min_distance = distance
                    closest_color = name
            
            return closest_color
            
        except Exception:
            return "Unknown"
    
    def color_distance(self, color1, color2):
        """
        Calculate Euclidean distance between two RGB colors
        
        Args:
            color1: First RGB tuple
            color2: Second RGB tuple
        
        Returns:
            Distance value
        """
        try:
            r1, g1, b1 = color1
            r2, g2, b2 = color2
            return ((r1-r2)**2 + (g1-g2)**2 + (b1-b2)**2)**0.5
        except Exception:
            return float('inf')
    
    def get_complementary_color(self, rgb):
        """
        Get complementary color
        
        Args:
            rgb: RGB tuple (r, g, b)
        
        Returns:
            Complementary RGB tuple
        """
        try:
            return (255 - rgb[0], 255 - rgb[1], 255 - rgb[2])
        except Exception:
            return (0, 0, 0)
    
    def get_analogous_colors(self, rgb, angle=30):
        """
        Get analogous colors (colors adjacent on color wheel)
        
        Args:
            rgb: RGB tuple (r, g, b)
            angle: Angle offset in degrees
        
        Returns:
            List of analogous RGB tuples
        """
        try:
            h, s, v = self.rgb_to_hsv(rgb)
            
            # Calculate analogous hues
            h1 = (h + angle) % 360
            h2 = (h - angle) % 360
            
            # Convert back to RGB
            rgb1 = self.hsv_to_rgb((h1, s, v))
            rgb2 = self.hsv_to_rgb((h2, s, v))
            
            return [rgb1, rgb2]
            
        except Exception:
            return [(0, 0, 0), (0, 0, 0)]
    
    def get_triadic_colors(self, rgb):
        """
        Get triadic colors (120 degrees apart on color wheel)
        
        Args:
            rgb: RGB tuple (r, g, b)
        
        Returns:
            List of triadic RGB tuples
        """
        try:
            h, s, v = self.rgb_to_hsv(rgb)
            
            # Calculate triadic hues
            h1 = (h + 120) % 360
            h2 = (h + 240) % 360
            
            # Convert back to RGB
            rgb1 = self.hsv_to_rgb((h1, s, v))
            rgb2 = self.hsv_to_rgb((h2, s, v))
            
            return [rgb1, rgb2]
            
        except Exception:
            return [(0, 0, 0), (0, 0, 0)]
    
    def get_color_brightness(self, rgb):
        """
        Calculate color brightness (perceived luminance)
        
        Args:
            rgb: RGB tuple (r, g, b)
        
        Returns:
            Brightness value (0-255)
        """
        try:
            r, g, b = rgb
            # Using standard luminance formula
            brightness = 0.299 * r + 0.587 * g + 0.114 * b
            return int(brightness)
        except Exception:
            return 0
    
    def is_dark_color(self, rgb, threshold=128):
        """
        Determine if a color is dark
        
        Args:
            rgb: RGB tuple (r, g, b)
            threshold: Brightness threshold
        
        Returns:
            Boolean indicating if color is dark
        """
        brightness = self.get_color_brightness(rgb)
        return brightness < threshold
    
    def get_contrast_color(self, rgb):
        """
        Get high contrast color (black or white)
        
        Args:
            rgb: RGB tuple (r, g, b)
        
        Returns:
            High contrast RGB tuple
        """
        if self.is_dark_color(rgb):
            return (255, 255, 255)  # White for dark colors
        else:
            return (0, 0, 0)  # Black for light colors
    
    def blend_colors(self, color1, color2, ratio=0.5):
        """
        Blend two colors
        
        Args:
            color1: First RGB tuple
            color2: Second RGB tuple
            ratio: Blend ratio (0.0 to 1.0)
        
        Returns:
            Blended RGB tuple
        """
        try:
            r1, g1, b1 = color1
            r2, g2, b2 = color2
            
            r = int(r1 * (1 - ratio) + r2 * ratio)
            g = int(g1 * (1 - ratio) + g2 * ratio)
            b = int(b1 * (1 - ratio) + b2 * ratio)
            
            return (r, g, b)
            
        except Exception:
            return (0, 0, 0)
    
    def get_color_temperature_name(self, temperature):
        """
        Get color temperature name
        
        Args:
            temperature: Temperature in Kelvin
        
        Returns:
            Temperature description string
        """
        if temperature < 3000:
            return "Very Warm"
        elif temperature < 4000:
            return "Warm"
        elif temperature < 5000:
            return "Neutral Warm"
        elif temperature < 6000:
            return "Neutral"
        elif temperature < 7000:
            return "Cool"
        else:
            return "Very Cool"
    
    def get_timestamp(self):
        """Get current timestamp"""
        return datetime.now().isoformat()
    
    def save_color_palette(self, colors, filename):
        """
        Save color palette to JSON file
        
        Args:
            colors: List of color dictionaries
            filename: Output filename
        """
        try:
            palette_data = {
                'colors': colors,
                'timestamp': self.get_timestamp(),
                'total_colors': len(colors)
            }
            
            with open(filename, 'w') as f:
                json.dump(palette_data, f, indent=2)
            
            print(f"[INFO] Color palette saved: {filename}")
            
        except Exception as e:
            print(f"[ERROR] Failed to save palette: {e}")
    
    def load_color_palette(self, filename):
        """
        Load color palette from JSON file
        
        Args:
            filename: Input filename
        
        Returns:
            List of color dictionaries
        """
        try:
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    palette_data = json.load(f)
                return palette_data.get('colors', [])
            else:
                print(f"[WARNING] Palette file not found: {filename}")
                return []
                
        except Exception as e:
            print(f"[ERROR] Failed to load palette: {e}")
            return []

def main():
    """Test color utilities"""
    utils = ColorUtils()
    
    # Test colors
    test_colors = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (128, 128, 128), # Gray
        (255, 165, 0),  # Orange
    ]
    
    print("[INFO] Testing Color Utilities:")
    print("=" * 50)
    
    for rgb in test_colors:
        hex_color = utils.rgb_to_hex(rgb)
        hsv = utils.rgb_to_hsv(rgb)
        cmyk = utils.rgb_to_cmyk(rgb)
        name = utils.get_color_name(rgb)
        brightness = utils.get_color_brightness(rgb)
        is_dark = utils.is_dark_color(rgb)
        
        print(f"RGB: {rgb}")
        print(f"  HEX: {hex_color}")
        print(f"  HSV: {hsv}")
        print(f"  CMYK: {cmyk}")
        print(f"  Name: {name}")
        print(f"  Brightness: {brightness}")
        print(f"  Is Dark: {is_dark}")
        print("-" * 30)

if __name__ == "__main__":
    main()

