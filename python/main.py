# Minecraft-Style World Generator - Phase 1: Basic Heightmap
# Requirements: pip install pygame numpy

import pygame
import numpy as np
import math
import hashlib
from pygame.locals import *

class PerlinNoise:
    """Simple Perlin noise implementation for terrain generation"""
    
    def __init__(self, seed=0):
        # Create permutation table based on seed
        np.random.seed(seed)
        self.perm = np.random.permutation(256)
        self.perm = np.concatenate([self.perm, self.perm])  # Duplicate for wraparound
        
    def fade(self, t):
        """Fade function for smooth interpolation"""
        return t * t * t * (t * (t * 6 - 15) + 10)
    
    def lerp(self, t, a, b):
        """Linear interpolation"""
        return a + t * (b - a)
    
    def grad(self, hash_val, x, y):
        """Gradient function"""
        h = hash_val & 3
        u = x if h < 2 else y
        v = y if h < 2 else x
        return (u if (h & 1) == 0 else -u) + (v if (h & 2) == 0 else -v)
    
    def noise(self, x, y):
        """Generate 2D Perlin noise at coordinates (x, y)"""
        # Find unit square containing point
        X = int(x) & 255
        Y = int(y) & 255
        
        # Relative position within square
        x -= int(x)
        y -= int(y)
        
        # Compute fade curves
        u = self.fade(x)
        v = self.fade(y)
        
        # Hash coordinates of square corners
        A = self.perm[X] + Y
        AA = self.perm[A]
        AB = self.perm[A + 1]
        B = self.perm[X + 1] + Y
        BA = self.perm[B]
        BB = self.perm[B + 1]
        
        # Blend results from 4 corners
        return self.lerp(v, 
                        self.lerp(u, self.grad(self.perm[AA], x, y),
                                    self.grad(self.perm[BA], x-1, y)),
                        self.lerp(u, self.grad(self.perm[AB], x, y-1),
                                    self.grad(self.perm[BB], x-1, y-1)))

class WorldGenerator:
    """Generates terrain using multiple octaves of Perlin noise"""
    
    def __init__(self, seed_string="default"):
        # Convert string seed to integer
        self.seed = int(hashlib.md5(seed_string.encode()).hexdigest()[:8], 16)
        self.noise = PerlinNoise(self.seed)
        
    def generate_heightmap(self, width, height, scale=50.0, octaves=4):
        """Generate a heightmap using fractional Brownian motion"""
        heightmap = np.zeros((height, width))
        
        for i in range(height):
            for j in range(width):
                # Normalize coordinates
                x = j / scale
                y = i / scale
                
                # Generate height using multiple octaves
                amplitude = 1.0
                frequency = 1.0
                max_value = 0.0
                height_value = 0.0
                
                for octave in range(octaves):
                    height_value += self.noise.noise(x * frequency, y * frequency) * amplitude
                    max_value += amplitude
                    amplitude *= 0.5
                    frequency *= 2.0
                
                # Normalize to [0, 1]
                heightmap[i][j] = (height_value / max_value + 1) / 2
                
        return heightmap

class TerrainRenderer:
    """Renders the terrain using Pygame"""
    
    def __init__(self, width=800, height=600):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Minecraft-Style World Generator")
        
    def heightmap_to_color(self, height):
        """Convert height value to color (blue=water, green=land, brown=mountain)"""
        if height < 0.3:
            # Water/beach - blue to light blue
            intensity = int(height * 255 / 0.3)
            return (intensity // 2, intensity // 2, 255)
        elif height < 0.6:
            # Grass/plains - green
            green_intensity = int(128 + (height - 0.3) * 127 / 0.3)
            return (34, green_intensity, 34)
        else:
            # Mountains - brown to white
            intensity = int(128 + (height - 0.6) * 127 / 0.4)
            return (intensity, intensity // 2, intensity // 4)
    
    def render_2d_heightmap(self, heightmap):
        """Render heightmap as 2D colored image"""
        h, w = heightmap.shape
        
        # Scale to fit screen
        scale_x = self.width // w
        scale_y = self.height // h
        
        self.screen.fill((0, 0, 0))
        
        for i in range(h):
            for j in range(w):
                height = heightmap[i][j]
                color = self.heightmap_to_color(height)
                
                # Draw scaled pixel
                pygame.draw.rect(self.screen, color, 
                               (j * scale_x, i * scale_y, scale_x, scale_y))
        
        # Add simple legend
        font = pygame.font.Font(None, 24)
        legend = [
            ("Blue: Water/Ocean", (100, 100, 255)),
            ("Green: Plains/Forest", (34, 200, 34)),
            ("Brown: Mountains", (139, 69, 19))
        ]
        
        for i, (text, color) in enumerate(legend):
            text_surface = font.render(text, True, color)
            self.screen.blit(text_surface, (10, 10 + i * 25))
        
        pygame.display.flip()
    
    def render_3d_wireframe(self, heightmap):
        """Simple 3D wireframe rendering"""
        h, w = heightmap.shape
        self.screen.fill((0, 0, 0))
        
        # Simple isometric projection
        offset_x = self.width // 2
        offset_y = self.height // 4
        scale = 2
        
        # Draw horizontal lines
        for i in range(0, h, 2):
            points = []
            for j in range(w):
                # Isometric projection
                x = (j - i) * scale + offset_x
                y = (j + i) * scale // 2 - heightmap[i][j] * 50 + offset_y
                points.append((x, y))
            
            if len(points) > 1:
                pygame.draw.lines(self.screen, (0, 255, 0), False, points, 1)
        
        # Draw vertical lines
        for j in range(0, w, 2):
            points = []
            for i in range(h):
                x = (j - i) * scale + offset_x
                y = (j + i) * scale // 2 - heightmap[i][j] * 50 + offset_y
                points.append((x, y))
            
            if len(points) > 1:
                pygame.draw.lines(self.screen, (0, 255, 0), False, points, 1)
        
        pygame.display.flip()

def main():
    """Main application loop"""
    print("=== Minecraft-Style World Generator ===")
    print("Enter a seed (or press Enter for random):")
    
    seed_input = input().strip()
    if not seed_input:
        import random
        seed_input = str(random.randint(1000, 9999))
        print(f"Using random seed: {seed_input}")
    
    # Generate world
    print("Generating world...")
    generator = WorldGenerator(seed_input)
    heightmap = generator.generate_heightmap(100, 100, scale=20.0)
    
    # Create renderer
    renderer = TerrainRenderer()
    
    print("\nControls:")
    print("SPACE - Toggle between 2D and 3D view")
    print("R - Regenerate with new random seed")
    print("ESC - Exit")
    
    # Main loop
    clock = pygame.time.Clock()
    running = True
    view_3d = False
    
    while running:
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    running = False
                elif event.key == K_SPACE:
                    view_3d = not view_3d
                elif event.key == K_r:
                    # Regenerate with new random seed
                    import random
                    new_seed = str(random.randint(1000, 9999))
                    print(f"Regenerating with seed: {new_seed}")
                    generator = WorldGenerator(new_seed)
                    heightmap = generator.generate_heightmap(100, 100, scale=20.0)
        
        # Render current view
        if view_3d:
            renderer.render_3d_wireframe(heightmap)
        else:
            renderer.render_2d_heightmap(heightmap)
        
        clock.tick(60)
    
    pygame.quit()

if __name__ == "__main__":
    main()