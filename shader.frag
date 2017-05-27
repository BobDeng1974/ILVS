#version 120

varying vec2 TexCoord0;
varying vec2 vertex0;
varying float Section0;

uniform sampler2D gSampler[3];
//uniform sampler2D gSampler1;
//uniform sampler2D gSampler2;
//uniform sampler2D gSampler3;



void main()
{
	float alpha = 0.5;
	float edge = 0.03;
	float slope = (1.0-alpha)/edge;
	float x = TexCoord0.x;
	float y = TexCoord0.y;
	
	vec4 tex1; //select corresponding texture
	if(Section0 == 0.0f){
		tex1 = texture2D(gSampler[0] , TexCoord0);
	}else if(Section0 == 1.0f){
		tex1 = texture2D(gSampler[1] , TexCoord0);
	}else if(Section0 == 2.0f){
		tex1 = texture2D(gSampler[2] , TexCoord0);
	}
	
	gl_FragColor.rgb = tex1.rgb;
	gl_FragColor.a = 1.0;

	//blending adjustment
	
	if( (x < edge || x > 1-edge) || (y < edge || y > 1-edge) ){
		if(x < edge){
			gl_FragColor.a = (slope*(x))/alpha;
		}else{
			gl_FragColor.a = (slope*(1-x))/alpha;
		}

		if(y < edge){
			gl_FragColor.a = (slope*(y))/alpha;
		}else if(y > 1-edge){
			gl_FragColor.a = (slope*(1-y))/alpha;
		}

	}
	
}
