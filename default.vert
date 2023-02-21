#version 460 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in int aType;

out vec3 color;

uniform mat4 view;
uniform mat4 projection;
uniform vec3 cameraEye;

uniform vec3 wireframeColor;
uniform vec3 color1;
uniform vec3 color2;
uniform vec3 color3;
uniform vec3 color4;
uniform vec3 color5;
uniform vec3 color6;

float pointSize = 10.0;

void main(){

	gl_Position = projection * view * vec4(aPos, 1.0f);
	float dist = distance(cameraEye, aPos);
	gl_PointSize = pointSize / (dist * dist);

	if(aType % 6 == 0){
		color = color1;
	}
	else if(aType % 6 == 1){
		 color = color2;
	}
	else if(aType % 6 == 2){
		 color = color3;
	}
	else if(aType % 6 == 3){
		 color = color4;
	}	
	else if(aType % 6 == 4){
		 color = color5;
	}	
	else if(aType % 6 == 5){
		 color = color6;
	}

	if(aType == 21){
		color = wireframeColor;
	}
}