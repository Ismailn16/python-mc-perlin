# Minecraft-Style World Generator - Improved Version
# Requirements: pip install pygame numpy

import pygame
import numpy as np
import math
import hashlib
import random
from pygame.locals import *

class ImprovedPerlinNoise:
    """Improved Perlin noise implementation with better gradients"""
    
    def __init__(self, seed=0):
        # Create permutation table based on seed
        np.random.seed(seed)
        self.perm = np.random.permutation(256)
        self.perm = np.concatenate([self.perm, self.perm])  # Duplicate for wraparound
        
        # Precompute gradients for better performance
        self.gradients = np.zeros((256, 2))
        for i in range(256):
            angle = 2 * np.pi * np.random.random()
            self.gradients[i] = [np.cos(angle), np.sin(angle)]
    
    def fade(self, t):
        """Improved fade function (6t^5 - 15t^4 + 10t^3)"""
        return t * t * t * (t * (t * 6 - 15) + 10)
    
    def lerp(self, t, a, b):
        """Linear interpolation"""
        return a + t * (b - a)
    
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
        
        # Calculate dot products
        gAA = self.gradients[AA]
        gBA = self.gradients[BA]
        gAB = self.gradients[AB]
        gBB = self.gradients[BB]
        
        # Calculate contributions from each corner
        a = self.lerp(u, gAA[0]*x + gAA[1]*y, gBA[0]*(x-1) + gBA[1]*y)
        b = self.lerp(u, gAB[0]*x + gAB[1]*(y-1), gBB[0]*(x-1) + gBB[1]*(y-1))
        
        # Blend results from 4 corners
        return self.lerp(v, a, b)

class WorldGenerator:
    """Generates terrain using multiple noise layers for realistic Minecraft-like terrain"""
    
    def __init__(self, seed_string="default"):
        # Convert string seed to integer
        self.seed = int(hashlib.md5(seed_string.encode()).hexdigest()[:8], 16)
        self.terrain_noise = ImprovedPerlinNoise(self.seed)
        self.biome_noise = ImprovedPerlinNoise(self.seed + 1)  # Different seed for biome map
        self.detail_noise = ImprovedPerlinNoise(self.seed + 2)  # For small details
        
    def generate_heightmap(self, width, height, scale=100.0, octaves=6):
        """Generate a heightmap using multiple noise layers"""
        heightmap = np.zeros((height, width))
        biome_map = np.zeros((height, width))
        
        # Generate base terrain with large features
        for i in range(height):
            for j in range(width):
                # Normalize coordinates
                x = j / scale
                y = i / scale
                
                # Generate base height using multiple octaves
                amplitude = 1.0
                frequency = 1.0
                max_value = 0.0
                height_value = 0.0
                
                for octave in range(octaves):
                    height_value += self.terrain_noise.noise(x * frequency, y * frequency) * amplitude
                    max_value += amplitude
                    amplitude *= 0.5
                    frequency *= 2.0
                
                # Normalize to [0, 1]
                base_height = (height_value / max_value + 1) / 2
                
                # Generate biome map (determines terrain type)
                biome_value = (self.biome_noise.noise(x * 0.5, y * 0.5) + 1) / 2
                biome_map[i][j] = biome_value
                
                # Add small details
                detail = self.detail_noise.noise(x * 4, y * 4) * 0.1
                
                # Apply terrain shaping based on biome
                if biome_value < 0.3:  # Ocean biome
                    height_value = base_height * 0.2 + detail
                elif biome_value < 0.5:  # Coastal/lowlands
                    height_value = base_height * 0.4 + 0.1 + detail
                elif biome_value < 0.7:  # Plains/forest
                    height_value = base_height * 0.6 + 0.2 + detail
                else:  # Mountains
                    # Make mountains more dramatic
                    mountain_height = base_height ** 2
                    height_value = mountain_height * 0.7 + 0.3 + detail
                
                heightmap[i][j] = height_value
        
        # Apply smoothing filter to reduce sharp transitions
        heightmap = self.smooth_heightmap(heightmap)
        
        return heightmap, biome_map
    
    def smooth_heightmap(self, heightmap):
        """Apply a simple smoothing filter to the heightmap"""
        kernel = np.array([[0.05, 0.1, 0.05],
                           [0.1, 0.4, 0.1],
                           [0.05, 0.1, 0.05]])
        
        smoothed = np.zeros_like(heightmap)
        h, w = heightmap.shape
        
        for i in range(1, h-1):
            for j in range(1, w-1):
                smoothed[i, j] = np.sum(heightmap[i-1:i+2, j-1:j+2] * kernel)
        
        return smoothed

class TerrainRenderer:
    """Renders the terrain using Pygame with improved visuals"""
    
    def __init__(self, width=800, height=600):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Improved Minecraft-Style World Generator")
        
        # Load block textures (simplified as colored squares for now)
        self.textures = {
            'water': pygame.Surface((16, 16)),
            'sand': pygame.Surface((16, 16)),
            'grass': pygame.Surface((16, 16)),
            'forest': pygame.Surface((16, 16)),
            'mountain': pygame.Surface((16, 16)),
            'snow': pygame.Surface((16, 16))
        }
        
        # Define colors for different terrain types
        self.textures['water'].fill((0, 0, 200))
        self.textures['sand'].fill((240, 230, 140))
        self.textures['grass'].fill((34, 139, 34))
        self.textures['forest'].fill((0, 100, 0))
        self.textures['mountain'].fill((139, 137, 137))
        self.textures['snow'].fill((255, 250, 250))
        
    def get_terrain_color(self, height, biome_value):
        """Convert height and biome value to appropriate color"""
        if height < 0.25:  # Deep water
            return (0, 0, 128)
        elif height < 0.3:  # Shallow water
            return (0, 0, 200)
        elif height < 0.35:  # Beach/sand
            return (240, 230, 140)
        elif height < 0.6:  # Grass/plains
            if biome_value < 0.5:
                return (34, 139, 34)  # Light green for plains
            else:
                return (0, 100, 0)  # Dark green for forests
        elif height < 0.8:  # Mountains
            return (139, 137, 137)  # Gray
        else:  # Snow caps
            return (255, 250, 250)  # White
    
    def render_2d_heightmap(self, heightmap, biome_map):
        """Render heightmap as 2D colored image with better scaling"""
        h, w = heightmap.shape
        
        # Scale to fit screen
        scale_x = self.width // w
        scale_y = self.height // h
        
        self.screen.fill((0, 0, 0))
        
        for i in range(h):
            for j in range(w):
                height = heightmap[i][j]
                biome = biome_map[i][j]
                color = self.get_terrain_color(height, biome)
                
                # Draw scaled pixel
                pygame.draw.rect(self.screen, color, 
                               (j * scale_x, i * scale_y, scale_x, scale_y))
        
        # Add improved legend
        font = pygame.font.Font(None, 24)
        legend = [
            ("Dark Blue: Deep Water", (0, 0, 128)),
            ("Blue: Shallow Water", (0, 0, 200)),
            ("Beige: Beach/Sand", (240, 230, 140)),
            ("Green: Plains", (34, 139, 34)),
            ("Dark Green: Forest", (0, 100, 0)),
            ("Gray: Mountains", (139, 137, 137)),
            ("White: Snow", (255, 250, 250))
        ]
        
        for i, (text, color) in enumerate(legend):
            text_surface = font.render(text, True, color)
            self.screen.blit(text_surface, (10, 10 + i * 25))
        
        pygame.display.flip()
    
    def render_3d_perspective(self, heightmap, biome_map):
        """Improved 3D perspective rendering"""
        h, w = heightmap.shape
        self.screen.fill((135, 206, 235))  # Sky blue background
        
        # 3D projection parameters
        scale = 10
        elevation = 30  # Viewing angle
        z_scale = 50   # Height exaggeration
        
        # Calculate center offset
        center_x = self.width // 2
        center_y = self.height // 2
        
        # Draw from back to front for simple hidden surface removal
        for i in range(h-1, -1, -1):
            for j in range(w):
                # Calculate 3D position
                x = (j - w/2) * scale
                y = (i - h/2) * scale
                z = heightmap[i][j] * z_scale
                
                # Apply simple perspective projection
                depth = 1 + (z / 200)  # Adjust for depth effect
                screen_x = int(center_x + x / depth)
                screen_y = int(center_y + y / depth - z / depth)
                
                # Calculate size based on depth
                size = max(1, int(scale / depth))
                
                # Get color based on height and biome
                color = self.get_terrain_color(heightmap[i][j], biome_map[i][j])
                
                # Draw the block
                if 0 <= screen_x < self.width and 0 <= screen_y < self.height:
                    pygame.draw.rect(self.screen, color, 
                                   (screen_x, screen_y, size, size))
        
        pygame.display.flip()

def main():
    """Main application loop"""
    print("=== Improved Minecraft-Style World Generator ===")
    print("Enter a seed (or press Enter for random):")
    
    seed_input = input().strip()
    if not seed_input:
        import random
        seed_input = str(random.randint(1000, 9999))
        print(f"Using random seed: {seed_input}")
    
    # Generate world
    print("Generating world...")
    generator = WorldGenerator(seed_input)
    heightmap, biome_map = generator.generate_heightmap(100, 100, scale=50.0)
    
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
                    print("Switched to", "3D" if view_3d else "2D", "view")
                elif event.key == K_r:
                    # Regenerate with new random seed
                    new_seed = str(random.randint(1000, 9999))
                    print(f"Regenerating with seed: {new_seed}")
                    generator = WorldGenerator(new_seed)
                    heightmap, biome_map = generator.generate_heightmap(100, 100, scale=50.0)
        
        # Render current view
        if view_3d:
            renderer.render_3d_perspective(heightmap, biome_map)
        else:
            renderer.render_2d_heightmap(heightmap, biome_map)
        
        clock.tick(30)
    
    pygame.quit()

if __name__ == "__main__":
    main()