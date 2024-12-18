#version 330 core
in vec2 tex_coords;
out vec4 FragColor;
uniform sampler2D textureSampler;

void main() {
    vec3 texColor = texture(textureSampler, tex_coords).rgb;
    FragColor = vec4(texColor, 1.0);
}