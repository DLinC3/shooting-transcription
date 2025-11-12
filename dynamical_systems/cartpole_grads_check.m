function cartpole_grads_check()
param.mc=1.0; param.mp=0.1; param.l=0.5; param.g=9.81; param.b=0.05; param.d=0.02;
eps = 1e-7;
for k=1:10
  x = [randn; (rand-0.5)*2*pi; randn; randn];  % random angle
  u = randn;
  % analytical gradient
  [Gx,Gu] = cartpole_grads(0,x,u,param);
  % finite difference
  f0 = cartpole_dynamics(0,x,u,param);
  Gx_fd = zeros(4,4); Gu_fd = zeros(4,1);
  for j=1:4
    xe=x; xe(j)=xe(j)+eps;
    xd=x; xd(j)=xd(j)-eps;
    Gx_fd(:,j) = (cartpole_dynamics(0,xe,u,param)-cartpole_dynamics(0,xd,u,param))/(2*eps);
  end
  Gu_fd = (cartpole_dynamics(0,x,u+eps,param)-cartpole_dynamics(0,x,u-eps,param))/(2*eps);
  % error
  ex = max(abs(Gx(:)-Gx_fd(:)));
  eu = max(abs(Gu(:)-Gu_fd(:)));
  fprintf('test %d: max|dfdx| err=%.3e, max|dfdu| err=%.3e\n',k,ex,eu);
end
end
