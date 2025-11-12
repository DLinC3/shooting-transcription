function xdot=cartpole_dynamics(t,x,u,param)

mc = param.mc;
mp = param.mp;
l = param.l;
g=param.g;
b=param.b;
d=param.d;

x1=x(1);
x2=x(2);
x3=x(3);
x4=x(4);

s = sin(x(2)); c = cos(x(2));

xdot(1,:)=x(3);
xdot(2,:)=x(4);
xdot(3,:) = [u-b*x3+d*x4*c/l+ mp*s*(l*x4^2 + g*c)]/[mc+mp*s^2];
xdot(4,:) =[-u*c+b*x3*c-d*(mc+mp)*x4/(mp*l) - mp*l*x4^2*c*s - (mc+mp)*g*s/(mp*l)]/[l*(mc+mp*s^2)];

end