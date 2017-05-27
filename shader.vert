#version 120

attribute vec3 Position;
attribute vec2 TexCoord;
attribute float Section;

varying vec2 TexCoord0;
varying vec2 vertex0;
varying float Section0;

void main()
{
   gl_Position = vec4(Position, 1.0);
   vertex0.xy = Position.xy;
   TexCoord0 = TexCoord;
   Section0 = Section;

}
