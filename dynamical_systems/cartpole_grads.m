function [dfdx dfdu]=cartpole_grads(t,x,u,param)

mc = param.mc;
mp = param.mp;
l = param.l;
g = param.g;
b = param.b;
d=param.d;

x1=x(1);
x2=x(2);
x3=x(3);
x4=x(4);

s=sin(x(2));
c=cos(x(2));


dfdx(1,1)=0;
dfdx(1,2)=0;
dfdx(1,3)=1;
dfdx(1,4)=0;

dfdx(2,1)=0;
dfdx(2,2)=0;
dfdx(2,3)=0;
dfdx(2,4)=1;

df3dx1 =0;
  
df3dx2 =- (g*mp*sin(x2)^2 - mp*cos(x2)*(l*x4^2 + g*cos(x2)) + (d*x4*sin(x2))/l)/(mp*sin(x2)^2 + mc) - (2*mp*cos(x2)*sin(x2)*(u - b*x3 + mp*sin(x2)*(l*x4^2 + g*cos(x2)) + (d*x4*cos(x2))/l))/(mp*sin(x2)^2 + mc)^2;
  
df3dx3 =-b/(mp*sin(x2)^2 + mc);
  
df3dx4 =((d*cos(x2))/l + 2*l*mp*x4*sin(x2))/(mp*sin(x2)^2 + mc);
  
df4dx1 =0;
  
df4dx2 =(2*mp*cos(x2)*sin(x2)*(l*mp*cos(x2)*sin(x2)*x4^2 + (d*(mc + mp)*x4)/(l*mp) + u*cos(x2) - b*x3*cos(x2) + (g*sin(x2)*(mc + mp))/(l*mp)))/(l*(mp*sin(x2)^2 + mc)^2) - (b*x3*sin(x2) - u*sin(x2) + l*mp*x4^2*cos(x2)^2 - l*mp*x4^2*sin(x2)^2 + (g*cos(x2)*(mc + mp))/(l*mp))/(l*(mp*sin(x2)^2 + mc));
  
df4dx3 =(b*cos(x2))/(l*(mc - mp*(cos(x2)^2 - 1)));
  
df4dx4 =-((d*(mc + mp))/(l*mp) + l*mp*x4*sin(2*x2))/(l*(mp*sin(x2)^2 + mc));
  
df3du =1/(mp*sin(x2)^2 + mc);
  
df4du =-cos(x2)/(l*(mc - mp*(cos(x2)^2 - 1)));

dfdx(3,1) =df3dx1;
dfdx(3,2) =df3dx2;
dfdx(3,3) =df3dx3;
dfdx(3,4) =df3dx4;

dfdx(4,1) =df4dx1;
dfdx(4,2) =df4dx2;
dfdx(4,3) =df4dx3;
dfdx(4,4) =df4dx4;

dfdu=[0;0;df3du;df4du];


end