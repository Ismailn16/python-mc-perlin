// Minecraft-Style World Generator - C++ OpenGL Version
// Build: g++ -o world_gen main.cpp -lglfw -lGL -lGLEW -std=c++17
// Dependencies: GLFW, GLEW, OpenGL

#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <cmath>
#include <functional>
#include <GL/glew.h>
#include <GLFW/glfw3.h>

// Simple 3D vector class
struct Vec3 {
    float x, y, z;
    Vec3(float x = 0, float y = 0, float z = 0) : x(x), y(y), z(z) {}
    Vec3 operator+(const Vec3& other) const { return Vec3(x + other.x, y + other.y, z + other.z); }
    Vec3 operator*(float scalar) const { return Vec3(x * scalar, y * scalar, z * scalar); }
};

// Vertex structure for OpenGL
struct Vertex {
    Vec3 position;
    Vec3 color;
    
    Vertex(const Vec3& pos, const Vec3& col) : position(pos), color(col) {}
};

class PerlinNoise {
private:
    std::vector<int> permutation;
    
    // Fade function for smooth interpolation
    float fade(float t) const {
        return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
    }
    
    // Linear interpolation
    float lerp(float t, float a, float b) const {
        return a + t * (b - a);
    }
    
    // Gradient function
    float grad(int hash, float x, float y) const {
        int h = hash & 3;
        float u = h < 2 ? x : y;
        float v = h < 2 ? y : x;
        return ((h & 1) == 0 ? u : -u) + ((h & 2) == 0 ? v : -v);
    }
    
public:
    PerlinNoise(unsigned int seed) {
        // Initialize permutation table
        permutation.resize(512);
        
        // Fill with values 0-255
        for (int i = 0; i < 256; i++) {
            permutation[i] = i;
        }
        
        // Shuffle using seed
        std::mt19937 rng(seed);
        std::shuffle(permutation.begin(), permutation.begin() + 256, rng);
        
        // Duplicate for wraparound
        for (int i = 0; i < 256; i++) {
            permutation[256 + i] = permutation[i];
        }
    }
    
    // Generate 2D Perlin noise
    float noise(float x, float y) const {
        // Find unit square containing point
        int X = static_cast<int>(std::floor(x)) & 255;
        int Y = static_cast<int>(std::floor(y)) & 255;
        
        // Relative position within square
        x -= std::floor(x);
        y -= std::floor(y);
        
        // Compute fade curves
        float u = fade(x);
        float v = fade(y);
        
        // Hash coordinates of square corners
        int A = permutation[X] + Y;
        int AA = permutation[A];
        int AB = permutation[A + 1];
        int B = permutation[X + 1] + Y;
        int BA = permutation[B];
        int BB = permutation[B + 1];
        
        // Blend results from 4 corners
        return lerp(v, 
                   lerp(u, grad(permutation[AA], x, y),
                           grad(permutation[BA], x - 1, y)),
                   lerp(u, grad(permutation[AB], x, y - 1),
                           grad(permutation[BB], x - 1, y - 1)));
    }
};

class WorldGenerator {
private:
    PerlinNoise noise;
    
public:
    WorldGenerator(const std::string& seedString) 
        : noise(stringToSeed(seedString)) {}
    
    // Convert string to numeric seed using hash
    unsigned int stringToSeed(const std::string& str) const {
        std::hash<std::string> hasher;
        return static_cast<unsigned int>(hasher(str));
    }
    
    // Generate heightmap using fractional Brownian motion
    std::vector<std::vector<float>> generateHeightmap(int width, int height, 
                                                     float scale = 50.0f, 
                                                     int octaves = 4) const {
        std::vector<std::vector<float>> heightmap(height, std::vector<float>(width));
        
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                float x = j / scale;
                float y = i / scale;
                
                float amplitude = 1.0f;
                float frequency = 1.0f;
                float maxValue = 0.0f;
                float heightValue = 0.0f;
                
                // Add multiple octaves
                for (int octave = 0; octave < octaves; octave++) {
                    heightValue += noise.noise(x * frequency, y * frequency) * amplitude;
                    maxValue += amplitude;
                    amplitude *= 0.5f;
                    frequency *= 2.0f;
                }
                
                // Normalize to [0, 1]
                heightmap[i][j] = (heightValue / maxValue + 1.0f) / 2.0f;
            }
        }
        
        return heightmap;
    }
};

class TerrainRenderer {
private:
    GLFWwindow* window;
    GLuint VAO, VBO, EBO;
    GLuint shaderProgram;
    
    // Camera properties
    Vec3 cameraPos{50.0f, 30.0f, 50.0f};
    Vec3 cameraTarget{0.0f, 0.0f, 0.0f};
    float cameraAngleX = 0.0f;
    float cameraAngleY = 0.0f;
    
    // Simple vertex shader source
    const char* vertexShaderSource = R"(
        #version 330 core
        layout (location = 0) in vec3 aPos;
        layout (location = 1) in vec3 aColor;
        
        out vec3 vertexColor;
        
        uniform mat4 view;
        uniform mat4 projection;
        
        void main() {
            gl_Position = projection * view * vec4(aPos, 1.0);
            vertexColor = aColor;
        }
    )";
    
    // Simple fragment shader source
    const char* fragmentShaderSource = R"(
        #version 330 core
        in vec3 vertexColor;
        out vec4 FragColor;
        
        void main() {
            FragColor = vec4(vertexColor, 1.0);
        }
    )";
    
    // Get terrain color based on height
    Vec3 heightToColor(float height) const {
        if (height < 0.3f) {
            // Water - blue
            float intensity = height / 0.3f;
            return Vec3(intensity * 0.5f, intensity * 0.5f, 1.0f);
        } else if (height < 0.6f) {
            // Grass - green
            float greenIntensity = 0.5f + (height - 0.3f) * 0.5f / 0.3f;
            return Vec3(0.1f, greenIntensity, 0.1f);
        } else {
            // Mountains - brown to white
            float intensity = 0.5f + (height - 0.6f) * 0.5f / 0.4f;
            return Vec3(intensity, intensity * 0.5f, intensity * 0.25f);
        }
    }
    
    // Compile shader
    GLuint compileShader(const char* source, GLenum type) {
        GLuint shader = glCreateShader(type);
        glShaderSource(shader, 1, &source, nullptr);
        glCompileShader(shader);
        
        // Check for compilation errors
        GLint success;
        glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
        if (!success) {
            char infoLog[512];
            glGetShaderInfoLog(shader, 512, nullptr, infoLog);
            std::cerr << "Shader compilation failed: " << infoLog << std::endl;
        }
        
        return shader;
    }
    
    // Create shader program
    void createShaderProgram() {
        GLuint vertexShader = compileShader(vertexShaderSource, GL_VERTEX_SHADER);
        GLuint fragmentShader = compileShader(fragmentShaderSource, GL_FRAGMENT_SHADER);
        
        shaderProgram = glCreateProgram();
        glAttachShader(shaderProgram, vertexShader);
        glAttachShader(shaderProgram, fragmentShader);
        glLinkProgram(shaderProgram);
        
        // Check for linking errors
        GLint success;
        glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
        if (!success) {
            char infoLog[512];
            glGetProgramInfoLog(shaderProgram, 512, nullptr, infoLog);
            std::cerr << "Shader linking failed: " << infoLog << std::endl;
        }
        
        glDeleteShader(vertexShader);
        glDeleteShader(fragmentShader);
    }
    
    // Simple matrix multiplication for view matrix
    void multiplyMatrix(float result[16], const float a[16], const float b[16]) {
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                result[i * 4 + j] = 0;
                for (int k = 0; k < 4; k++) {
                    result[i * 4 + j] += a[i * 4 + k] * b[k * 4 + j];
                }
            }
        }
    }
    
public:
    TerrainRenderer(int width = 1024, int height = 768) {
        // Initialize GLFW
        glfwInit();
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        
        // Create window
        window = glfwCreateWindow(width, height, "Minecraft-Style World Generator (C++)", nullptr, nullptr);
        glfwMakeContextCurrent(window);
        
        // Initialize GLEW
        glewInit();
        
        // Enable depth testing
        glEnable(GL_DEPTH_TEST);
        
        // Create shader program
        createShaderProgram();
        
        // Generate VAO and VBO
        glGenVertexArrays(1, &VAO);
        glGenBuffers(1, &VBO);
        glGenBuffers(1, &EBO);
    }
    
    ~TerrainRenderer() {
        glDeleteVertexArrays(1, &VAO);
        glDeleteBuffers(1, &VBO);
        glDeleteBuffers(1, &EBO);
        glDeleteProgram(shaderProgram);
        glfwTerminate();
    }
    
    // Generate mesh from heightmap
    void generateMesh(const std::vector<std::vector<float>>& heightmap) {
        std::vector<Vertex> vertices;
        std::vector<unsigned int> indices;
        
        int height = heightmap.size();
        int width = heightmap[0].size();
        
        // Generate vertices
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                float h = heightmap[i][j];
                Vec3 position(j, h * 20.0f, i);  // Scale height
                Vec3 color = heightToColor(h);
                vertices.emplace_back(position, color);
            }
        }
        
        // Generate indices for triangles
        for (int i = 0; i < height - 1; i++) {
            for (int j = 0; j < width - 1; j++) {
                int topLeft = i * width + j;
                int topRight = i * width + (j + 1);
                int bottomLeft = (i + 1) * width + j;
                int bottomRight = (i + 1) * width + (j + 1);
                
                // First triangle
                indices.push_back(topLeft);
                indices.push_back(bottomLeft);
                indices.push_back(topRight);
                
                // Second triangle
                indices.push_back(topRight);
                indices.push_back(bottomLeft);
                indices.push_back(bottomRight);
            }
        }
        
        // Upload to GPU
        glBindVertexArray(VAO);
        
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Vertex), 
                    vertices.data(), GL_STATIC_DRAW);
        
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), 
                    indices.data(), GL_STATIC_DRAW);
        
        // Position attribute
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);
        glEnableVertexAttribArray(0);
        
        // Color attribute
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), 
                             (void*)offsetof(Vertex, color));
        glEnableVertexAttribArray(1);
    }
    
    // Update camera based on input
    void updateCamera() {
        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) cameraAngleX -= 0.02f;
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) cameraAngleX += 0.02f;
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) cameraAngleY -= 0.02f;
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) cameraAngleY += 0.02f;
        
        // Calculate camera position
        float radius = 100.0f;
        cameraPos.x = radius * std::cos(cameraAngleY) * std::cos(cameraAngleX);
        cameraPos.y = radius * std::sin(cameraAngleX) + 30.0f;
        cameraPos.z = radius * std::sin(cameraAngleY) * std::cos(cameraAngleX);
    }
    
    // Render the terrain
    void render() {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glClearColor(0.6f, 0.8f, 1.0f, 1.0f);  // Sky blue background
        
        updateCamera();
        
        glUseProgram(shaderProgram);
        
        // Create view matrix (simple look-at)
        float viewMatrix[16] = {
            1, 0, 0, -cameraPos.x,
            0, 1, 0, -cameraPos.y,
            0, 0, 1, -cameraPos.z,
            0, 0, 0, 1
        };
        
        // Create projection matrix (simple perspective)
        float fov = 45.0f * M_PI / 180.0f;
        float aspect = 1024.0f / 768.0f;
        float near = 0.1f;
        float far = 1000.0f;
        
        float f = 1.0f / std::tan(fov / 2.0f);
        float projectionMatrix[16] = {
            f/aspect, 0, 0, 0,
            0, f, 0, 0,
            0, 0, (far+near)/(near-far), (2*far*near)/(near-far),
            0, 0, -1, 0
        };
        
        // Upload matrices to shader
        GLint viewLoc = glGetUniformLocation(shaderProgram, "view");
        GLint projLoc = glGetUniformLocation(shaderProgram, "projection");
        glUniformMatrix4fv(viewLoc, 1, GL_FALSE, viewMatrix);
        glUniformMatrix4fv(projLoc, 1, GL_FALSE, projectionMatrix);
        
        // Draw terrain
        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES, 99*99*6, GL_UNSIGNED_INT, 0);  // Assuming 100x100 heightmap
        
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    
    bool shouldClose() const {
        return glfwWindowShouldClose(window);
    }
};

int main() {
    std::cout << "=== Minecraft-Style World Generator (C++) ===" << std::endl;
    std::cout << "Enter a seed (or press Enter for 'default'): ";
    
    std::string seedInput;
    std::getline(std::cin, seedInput);
    if (seedInput.empty()) {
        seedInput = "default";
    }
    
    std::cout << "Generating world with seed: " << seedInput << std::endl;
    
    // Generate world
    WorldGenerator generator(seedInput);
    auto heightmap = generator.generateHeightmap(100, 100, 20.0f, 4);
    
    // Create renderer and generate mesh
    TerrainRenderer renderer;
    renderer.generateMesh(heightmap);
    
    std::cout << "\nControls:" << std::endl;
    std::cout << "WASD - Rotate camera" << std::endl;
    std::cout << "ESC - Exit" << std::endl;
    
    // Main render loop
    while (!renderer.shouldClose()) {
        if (glfwGetKey(glfwGetCurrentContext(), GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            break;
        }
        
        renderer.render();
    }
    
    return 0;
}