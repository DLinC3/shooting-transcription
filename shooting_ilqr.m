function shooting_ilqr

    T = 2.5; %time horizon
    dt = 1/60;%time step
    N = floor(T/dt);% number of time steps
    nX = 4;%number of states
    nU = 1;%number of inputs
    
    t = zeros(1,N);%time
    x = zeros(nX,N);%state
    u = zeros(nU,N);%input
    
    %cartpole physical parameters
    param.mc = 10; 
    param.mp = 2; 
    param.l = 0.5; 
    param.g = 9.8;
    param.b=0.1;
    param.d=0.1;
    
    %initial conditions
    x0= [0;0;0;0];
    
    xtraj=zeros(nX,N);%state-trajectory
    utraj=zeros(nU,N);%input-trajectory
    ktraj= zeros(nU,N);%gain-trajectory
    Ktraj = zeros(nU,nX,N);%gain-trajectory

    % fill
    % For the case where the angular velocity at the vertex is too high:
    % 
    Q = diag([0.1, 0.01, 0.01, 0.01]);% cost
    Qf = diag([10, 500, 1, 2]);% final cost
    R = 0.001;% input cost

    xd = [0;pi;0;0];%desired state
    
    %call iterative lqr
    [xtraj, utraj, ktraj, Ktraj] =ilqr(x0, xtraj, utraj, ktraj, Ktraj,N,dt,param,Q, R, Qf,xd);
    
    x(:,1)=[0;0;0;0];
    
    %simulate the cart-pole system
    for k =1:N-1
        x(:,k+1)=x(:,k)+ (cartpole_dynamics(t(k),x(:,k),utraj(:,k),param))*dt;
        t(k+1) = t(k)+dt;
    end
    
    %draw the cartpole system
    for k =1:N
        draw_cartpole(t(k),x(:,k),param);
        drawnow;
    end  
    x(:,N)
    %-------------------------------
    % Animate and record video
    %-------------------------------
    fprintf('Animating solution...\n');
    figure(25); clf;
    
    v = VideoWriter('ilqr.mp4', 'MPEG-4');
    v.FrameRate = round(1/dt);  
    v.Quality   = 100;   
    open(v);
    
    for k = 1:N
        draw_cartpole(t(k), x(:,k), param); 
        frame = getframe(gcf); 
        writeVideo(v, frame); 
        drawnow;
    end
    
    close(v);
    fprintf('Video saved as ilqr.mp4\n');
end

%==========================================================================
%   iterativeLQR function
%==========================================================================
function [xtraj, utraj, ktraj, Ktraj] = ilqr(x0, xtraj, utraj, ktraj, Ktraj,N,dt,param, Q, R, Qf,xd)
    J=1e6;
    Jlast = J;
    for i=1:1000
        [xtraj, utraj, J]=forward_pass(x0, xtraj, utraj, ktraj, Ktraj, N, dt,param,Q,R,Qf,xd,J);
        [Ktraj, ktraj]=backward_pass(xtraj, utraj, ktraj, Ktraj, Q, R, Qf, xd, param,N,dt);
        figure(1)
        plot(xtraj(1,:),xtraj(2,:))
        if abs(J-Jlast) < 0.0001
            break;
        end
        Jlast = J;
    end
        figure(1)
        plot(xtraj(1,:),xtraj(3,:))
        figure(2)
        plot(xtraj(2,:),xtraj(4,:))
end

%==========================================================================
%   get Q-terms 
%==========================================================================
function [Qx, Qu, Qxx, Qux, Quu]= Q_terms (gx, gu, gxx, gux,guu, fx, fu, Vx, Vxx)
    
    % fill in
    Qx = gx + fx' * Vx;
    Qu = gu + fu' * Vx;
    Qxx = gxx + fx' * Vxx * fx;
    Qux = gux + fu' * Vxx * fx;
    Quu = guu + fu' * Vxx * fu;
end

%==========================================================================
%   get gains 
%==========================================================================
function [K, v]= gains(Qx,Qu,Qxx,Qux,Quu)
    v = -inv(Quu) * Qu;
    K = -inv(Quu) * Qux;
end

%==========================================================================
%   get V terms 
%==========================================================================
function [Vx, Vxx] = Vterms (Qx,Qu,Qxx,Qux,Quu,K,k)
    Vx = Qx - Qux' * inv(Quu) * Qu;
    Vxx = Qxx - Qux' * inv(Quu) * Qux;
end

%==========================================================================
%   compute backward pass
%==========================================================================
function [Ktraj, ktraj] = backward_pass(xtraj, utraj, ktraj, Ktraj, Q, R, Qf, xd,param,N,dt)
   
    % fill in
    % Set terminal value function derivatives using the final cost:
    Vxx = Qf;                    % Vxx,N = Qf
    Vx = Qf * (xtraj(:,N) - xd);  % Vx,N = Qf*(x_N - xd)

    for i=N-1:-1:1
        [gx, gu, gxx, gux, guu] = cost_gradients(xtraj(:,i),utraj(:,i),xd,Q,R);
        
        %fill in cartpole gradients
        % Compute continuous-time Jacobians at time step i:
        [fx_cont, fu_cont] = cartpole_grads(0, xtraj(:,i), utraj(:,i), param);

        fu = dt * fu_cont;
        fx = eye(length(xtraj(:,i))) + dt * fx_cont;

        [Qx, Qu, Qxx, Qux, Quu]= Q_terms (gx, gu, gxx, gux, guu, fx, fu, Vx, Vxx);
        [Ktraj(:,:,i), ktraj(:,i)]= gains(Qx,Qu,Qxx,Qux,Quu);
        [Vx, Vxx] = Vterms (Qx,Qu,Qxx,Qux,Quu,Ktraj(:,:,i), ktraj(:,i));
    end
end

%==========================================================================
%   cost function
%==========================================================================
function J = cost(x,u,Q,R)

J = 0.5*x'*Q*x + 0.5*u'*R*u;

end

%==========================================================================
%   final cost function
%==========================================================================
function Jf = final_cost(x,u,xd,Qf)
Jf = 0.5*(x-xd)'*Qf*(x-xd);
end

%==========================================================================
%   final cost gradients 
%==========================================================================
function [gx, gu, gxx, gux, guu] = final_cost_gradients(x,u,xd,Qf)
    gx = Qf * (x - xd);
    gu = zeros(size(u));
    gxx = Qf;
    gux = zeros(size(u,1), length(x));
    guu = zeros(size(u,1), size(u,1));
end

%==========================================================================
%   cost gradients 
%==========================================================================
function [gx, gu, gxx, gux, guu] = cost_gradients(x,u,xd,Q,R)
    gx = Q * x;
    gu = R * u;
    gxx = Q;
    gux = zeros(size(u,1), length(x)); % no cross–terms
    guu = R;
end

%==========================================================================
%   forward pass
%==========================================================================
function [xtraj, utraj, J] = forward_pass(x0, xtraj0, utraj0, ktraj, Ktraj, N, dt,param,Q,R,Qf,xd,J0)
    J = 1e7;
    alpha = 10;
    while(J0<J)
        t = 0;
        x = x0;
        J = 0;
        for i=1:N-1
            xtraj(:,i) = x;
            
            % fill in
             u = utraj0(:,i) + alpha * ktraj(:,i) + Ktraj(:,:,i) * (x - xtraj0(:,i));

            utraj(:,i) = u;%save forward pass input
            J = J+cost(x,utraj(:,i),Q,R);%compute cost of current iteration
            xdot = cartpole_dynamics(t,x,utraj(:,i),param);%compute cartpole derivatives
            t = t + dt;%increment time step
            x=x+xdot*dt;%integrate state    
        end
        xtraj(:,N) = x;
        J = J+final_cost(x,u,xd,Qf)%compute total cost
        J0
        alpha = alpha/2% linesearch if cost wasn't minimized
    end
end

%==========================================================================
%   cart pole dynamics
%==========================================================================
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

%==========================================================================
%   cart pole gradients
%==========================================================================
function [dfdx, dfdu]=cartpole_grads(t,x,u,param)

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

%==========================================================================
%   draw the cart-pole
%==========================================================================
function draw_cartpole(t,x,param)
      l=param.l;
      persistent hFig base a1 raarm wb lwheel rwheel;
      if (isempty(hFig))
        hFig = figure(25);
        set(hFig,'DoubleBuffer', 'on');
        
        a1 = l+0.25;
        av = pi*[0:.05:1];
        theta = pi*[0:0.05:2];
        wb = .3; hb=.15;
        aw = .01;
        wheelr = 0.05;
        lwheel = [-wb/2 + wheelr*cos(theta); -hb-wheelr + wheelr*sin(theta)]';
        base = [wb*[1 -1 -1 1]; hb*[1 1 -1 -1]]';
        arm = [aw*cos(av-pi/2) -a1+aw*cos(av+pi/2)
          aw*sin(av-pi/2) aw*sin(av+pi/2)]';
        raarm = [(arm(:,1).^2+arm(:,2).^2).^.5, atan2(arm(:,2),arm(:,1))];
      end
      
      figure(hFig); cla; hold on; view(0,90);
      patch(x(1)+base(:,1), base(:,2),0*base(:,1),'b','FaceColor',[.3 .6 .4])
      patch(x(1)+lwheel(:,1), lwheel(:,2), 0*lwheel(:,1),'k');
      patch(x(1)+wb+lwheel(:,1), lwheel(:,2), 0*lwheel(:,1),'k');
      patch(x(1)+raarm(:,1).*sin(raarm(:,2)+x(2)-pi),-raarm(:,1).*cos(raarm(:,2)+x(2)-pi), 1+0*raarm(:,1),'r','FaceColor',[.9 .1 0])
      plot3(x(1)+l*sin(x(2)), -l*cos(x(2)),1, 'ko',...
        'MarkerSize',10,'MarkerFaceColor','b')
      plot3(x(1),0,1.5,'k.')
      title(['t = ', num2str(t,'%.2f') ' sec']);
      set(gca,'XTick',[],'YTick',[])
      
      axis image;
      axis([-2.5 2.5 -2.5*l 2.5*l]);
      drawnow;
    end