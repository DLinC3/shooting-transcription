% =========================================================================
% Cartpole Backpropagation Through Time Trajectory Optimization
% =========================================================================
function  shooting_backpropogation()

close all
clear all

% set up the dynamics
addpath("dynamical_systems/")
param=init_cartpole_params();
GRADIENTS=@cartpole_grads;
DYNAMICS=@cartpole_dynamics;
DRAW=@draw_cartpole;
rmpath("dynamical_systems/")

param.dt=1/20; %time step

param.x0=[0 0 0 0]'; % initial conditions
param.xd = [0 pi 0 0]'; % final desired state
param.iter=200; % number of iterations


param.Q=diag([0.01, 0.01, 0.01, 0.01]); % Cost on glider trajectory
param.R=0.001; % actuator cost

param.Qend=diag([10, 500, 1, 50]); % final value cost

param.eta=0.005; % optimization update parameter
param.x0=[0 0 0 0]';

%bounds for plotting
param.xL=-2;
param.xU=2;
param.yL=-2*pi;
param.yU=2*pi;

dt=param.dt;%time step

N=50;%number of time steps
param.N=N;

utape0=0*ones(1,N);%initialize the initial open loop policy

ti=0;
t=ti:dt:(N-1)*dt;%time values

x0=param.x0;%set the initial condition

utape=bptt(utape0,x0,DYNAMICS,GRADIENTS,param);% run the bptt algorithm     

%simulate the dynamics
x=x0;
for k=1:N
    xtape(:,k)=x;
    xdot=DYNAMICS(t,x,utape(k),param);
    x=x+xdot*dt;
end
disp(xtape(:,N))
%plot the resulting trajectory             
figure()
plot(xtape(1,:),xtape(2,:))
axis([param.xL param.xU param.yL param.yU])
figure()
plot(utape);

% draw the robot
for i=1:size(xtape,2)
    DRAW((i-1)*dt,xtape(:,i),param);
end
fprintf('Animating solution...\n');
fig = figure(25); clf; set(fig,'Color','w');

v = VideoWriter('bptt_cartpole.mp4','MPEG-4');
v.FrameRate = round(1/param.dt);   % 与仿真一致
v.Quality   = 100;
open(v);

for i = 1:size(xtape,2)
    DRAW((i-1)*param.dt, xtape(:,i), param);
    drawnow limitrate;
    frame = getframe(gcf);
    writeVideo(v, frame);
end
close(v);

end

% =========================================================================
% This trajectory optimization method using back propagation through time 
% to calculate the gradients and gradient descent to optimize the
% trajectory
% =========================================================================
function utape = bptt(utape0,x,DYNAMICS,GRADIENTS,param)
    utape = gradient_descent(utape0,x,DYNAMICS,GRADIENTS,param);
end


% =========================================================================
% This function implements gradient descent optimization
% =========================================================================
function utape = gradient_descent(utape0,x,DYNAMICS,GRADIENTS,param)

x0=x; % initial state position
N=param.N;  % number of discrete time steps

J_old=100000; % initial cost
utape_old=utape0; % initial utape
utape=utape0; 
eta=param.eta; % gradient descent optimization gain

    for k=1:param.iter 
    
    % compute cost function gradients
    [J,dJdalpha] = bptt_gradients(utape0,x0,DYNAMICS,GRADIENTS,param);
    
    % compare current cost to previous cost
    
    if (J > J_old) % if current cost is greater, reduce optimization gain
     
        J = J_old;
        utape = utape_old;
        eta = eta/2;
           
    else % if current cost is less, increase the optimization gain
        eta = 1.1*eta;
    end
    
    % save last utape and cost
    utape_old = utape; 
    J_old = J;
    % disp(['utape0 size: ', num2str(size(utape0))]);
    % update input tape through gradient descent
    utape=utape0-eta*dJdalpha';
    
    utape0=utape;
    % disp(['utape size: ', num2str(size(utape))]); 
    
    end

end

% =========================================================================
% This function uses backprop to compute the gradients
% alpha reperesents the parameters for the open-loop policy. In this case
% alpha represents a control input at each discrete time-step
% =========================================================================

function [J,dJdalpha] = bptt_gradients(alpha,x0,DYNAMICS,GRADIENTS,param)

dt=param.dt; % discrete time interval
xd=param.xd; % final goal state

% construct time vector
ti=0;
N=length(alpha);
T=ti:dt:(N-1)*dt;
tf=T(N);
utape=alpha;

x=x0;
xtape = zeros(4, N);
for k=1:N
    % roll out the dynamics
    xtape(:,k) = x;  % store current state
    xdot = DYNAMICS(T(k), x, utape(k), param); 
    x = x + xdot*dt; % euler integrate
end


% plot first state against second state
figure(24)
plot(xtape(1,:),xtape(2,:),xd(1),xd(2),'o',x0(1),x0(2),'*'); 
xlabel('x-coordinate (m)');
ylabel('z-coordinate (m)');
title ('Real-time Planning');
axis([param.xL param.xU param.yL param.yU]);drawnow;

% compute the gradients of the cost function using backpropagation 
dJdalpha = compute_gradients(xtape,utape,N,param,GRADIENTS);

% compute the cost
J=cost(xtape,utape,param) + finalCost(xtape(:,N),param);

end

% =========================================================
% This function returns the gradients by
% integrating the adjoint equations
% =========================================================
function dJdalpha = compute_gradients(x,u,N,param,GRADIENTS)

dt=param.dt;
Q=param.Q;
R=param.R;
Qend=param.Qend;
xd=param.xd;

Q = Q.*dt;
R = R.*dt;


[dfdx,dfdu] = GRADIENTS(0,x(:,N),u(:,N),param);% dfdu 4*1 % dfdx 4*4


dgdu = 2*u(:,N)'*R; % 1*1 ∂g/∂u_N
dudalpha = 0*u; 
dudalpha(:,N)=1; %gradient of u w.r.t. parameters for open loop policy at N

F_alpha = dfdu*dudalpha;% F_alpha 4*50 % F derivitive w.r.t entire u
G_alpha = dgdu*dudalpha;% G_alpha 1*50

% λ_{N-1}^T = ∂g_f/∂x_N
% return 4*1
lambda = (2*(x(:,N) - xd)' * Qend)'; % give terminal condition for λ

% dL/du = ∂g/∂u_N + λ_{N-1}^T * ∂f/∂u_N
% dJdalpha should return 50*1
% so take the transpose
dJdalpha = (G_alpha + lambda' * F_alpha)' ; % dJdalpha for first time step


for n=N-1:-1:1 %integrate adjoint equations backwards in time
% g(x,u)= 1/2*(x^TQx + u^TRu)
% ∂g/∂x = x^TQ  1*4
% ∂g/∂u = u^TR  1*1
dgdx = x(:,n)'*Q; %gradient of cost with respect to current state 
dgdu = u(:,n)'*R; %gradient of cost with respect to current action

[dfdx,dfdu] = GRADIENTS(0,x(:,n),u(:,n),param); %gradient of f w.r.t. current position, action
F_x = dfdx; % 4*4
G_x = dgdx; % 1*4

% dudalpha 1*50
dudalpha = 0*dudalpha(:,1:N); dudalpha(:,n) = 1; %gradient of u w.r.t. parameters for open loop policy at current time


F_alpha = dfdu*dudalpha; % 4*50
G_alpha = dgdu*dudalpha; % 1*50


% λ_{n-1}^T = ∂g/∂x_n + λ_n^T * ∂f/∂x_n.
% however we are in the discretized simulation
lambda = lambda + (G_x + lambda'*F_x)'*dt; %solve for λ for previous step

% dL/du = ∂g/∂u_k + λ_{k}^T * ∂f/∂u_k
% dJdalpha should return 50*1
% since utape=utape0-eta*dJdalpha'
dJdalpha = dJdalpha + (G_alpha + lambda'* F_alpha)' ;%add this step's contribution to dJdalpha

end


end

% =============================================================
% This function defines the instantaneous cost (i.e. g(x,u))
% =============================================================
function J = cost(X,u,param)

R=param.R;
Q=param.Q;
dt=param.dt;

%%implement the discrete time running cost
J = 0;
for k = 1:size(X, 2)-1
    J = J + (X(:,k)' * Q * X(:,k) +  u(k) * R * u(k)) * dt;
end
end

% =============================================================
% Implements a final cost
% =============================================================
function Jf = finalCost(X,param)

xd=param.xd;
Qend=param.Qend;

%5implement the final cost
Jf = (X - xd)' * Qend * (X - xd);
end
