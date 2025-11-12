function shooting_DDP()

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

    % Cost weights
    Q = diag([0.1, 0.01, 0.01, 0.01]);% state cost
    Qf = diag([10, 500, 1, 2]);% final cost
    R = 0.001;% input cost

    xd = [0;pi;0;0];%desired state
    
    %call TRUE DDP
    [xtraj, utraj, ktraj, Ktraj] = true_ddp(x0, xtraj, utraj, ktraj, Ktraj,N,dt,param,Q, R, Qf,xd);
    
    x(:,1)=[0;0;0;0];
    
    %simulate the cart-pole system with final controller
    for k =1:N-1
        % Use feedback controller from DDP
        u_fb = utraj(:,k) + Ktraj(:,:,k) * (x(:,k) - xtraj(:,k));
        x(:,k+1)=x(:,k)+ (cartpole_dynamics(t(k),x(:,k),u_fb,param))*dt;
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
    
    v = VideoWriter('ddp.mp4', 'MPEG-4');
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
    fprintf('Video saved as ddp.mp4\n');


end

%==========================================================================
%   TRUE DDP main algorithm
%==========================================================================
function [xtraj, utraj, ktraj, Ktraj] = true_ddp(x0, xtraj, utraj, ktraj, Ktraj,N,dt,param, Q, R, Qf,xd)
    
    % DDP parameters with improved settings
    mu = 1.0;  % Start with larger regularization for stability
    mu_min = 1e-8;
    mu_max = 1e10;
    delta = 1.6;  % Gentler scaling factor (was 2)
    delta_0 = 2;  % For cost decrease acceptance
    max_iter = 1000;
    tol = 1e-4;
    
    % CRITICAL: Initialize with a nominal rollout before first backward pass
    % This ensures consistency between xtraj and utraj
    [xtraj, utraj, J] = nominal_rollout(x0, N, dt, param, Q, R, Qf, xd);
    
    % Track regularization history for adaptive adjustment
    mu_factor = 1.0;
    accept_count = 0;
    
    Jlast = J;
    
    for iter = 1:max_iter
        % Backward pass with regularization
        [Ktraj_new, ktraj_new, success] = backward_pass_true_ddp(xtraj, utraj, Q, R, Qf, xd, param, N, dt, mu);
        
        if ~success
            % If backward pass failed, increase regularization
            mu = min(mu * delta, mu_max);
            fprintf('Iter %d: Backward pass failed, increasing mu to %e\n', iter, mu);
            continue;
        end
        
        % Forward pass with line search
        [xtraj_new, utraj_new, J_new] = forward_pass_ddp(x0, xtraj, utraj, ktraj_new, Ktraj_new, N, dt, param, Q, R, Qf, xd);
        
        % Check if cost decreased
        if J_new < J
            % Accept the step
            xtraj = xtraj_new;
            utraj = utraj_new;
            ktraj = ktraj_new;
            Ktraj = Ktraj_new;
            
            % Update cost
            dJ = J - J_new;
            J = J_new;
            
            % Decrease regularization for next iteration
            mu = max(mu / delta, mu_min);
            
            iter
            J
            dJ
            mu
            
            % Visualize progress
            figure(1); clf;
            subplot(2,1,1);
            plot(xtraj(2,:)*180/pi);
            ylabel('Angle');
            grid on;
            
            subplot(2,1,2);
            plot(utraj);
            ylabel('Control');
            xlabel('Time step');
            grid on;
            drawnow;
            
            % Check convergence
            if abs(dJ) < tol
                iter
                break;
            end
        else
            % Reject step, increase regularization
            mu = min(mu * delta, mu_max);
            fprintf('Iter %d: Cost increased, rejecting step, mu = %e\n', iter, mu);
        end
    end
    
    if iter == max_iter
        fprintf('Maximum iterations reached\n');
    end
end

%==========================================================================
%   Nominal rollout for initialization with better initial guess
%==========================================================================
function [xtraj, utraj, J] = nominal_rollout(x0, N, dt, param, Q, R, Qf, xd)
    nX = 4;
    nU = 1;
    
    xtraj = zeros(nX, N);
    utraj = zeros(nU, N);
    
    % Start from initial state
    xtraj(:,1) = x0;
    J = 0;
    
    % Better initial control strategy: energy-based swing-up
    for k = 1:N-1
        x = xtraj(:,k);
        theta = x(2);
        theta_dot = x(4);
        
        % Simple energy-based swing-up controller
        E = 0.5 * param.mp * param.l^2 * theta_dot^2 - param.mp * param.g * param.l * cos(theta);
        E_desired = param.mp * param.g * param.l;  % Energy at upright position
        
        % Control law that pumps energy
        if abs(theta - pi) > 0.2  % Not near target
            k_E = 5.0;  % Energy gain
            utraj(:,k) = k_E * sign(theta_dot * cos(theta)) * (E - E_desired);
        else
            % Near target, use simple PD control
            k_p = 10;
            k_d = 2;
            utraj(:,k) = -k_p * (theta - pi) - k_d * theta_dot;
        end
        
        % Saturate control
        utraj(:,k) = max(min(utraj(:,k), 50), -50);
        
        % Compute cost
        J = J + stage_cost(xtraj(:,k), utraj(:,k), Q, R);
        
        % Simulate forward
        xdot = cartpole_dynamics(0, xtraj(:,k), utraj(:,k), param);
        xtraj(:,k+1) = xtraj(:,k) + dt * xdot;
    end
    
    % Add terminal cost
    J = J + terminal_cost(xtraj(:,N), xd, Qf);
end

%==========================================================================
%   TRUE DDP backward pass with improved numerics
%==========================================================================
function [Ktraj, ktraj, success, expected_improvement] = backward_pass_true_ddp(xtraj, utraj, Q, R, Qf, xd, param, N, dt, mu)
    nX = 4;
    nU = 1;
    
    success = true;
    Ktraj = zeros(nU, nX, N);
    ktraj = zeros(nU, N);
    
    % Set terminal value function derivatives
    Vxx = Qf;
    Vx = Qf * (xtraj(:,N) - xd);
    
    % Expected cost reduction (for line search)
    expected_improvement = 0;
    
    for k = N-1:-1:1
        % Get cost gradients (including second-order terms)
        [gx, gu, gxx, gux, guu] = cost_gradients_ddp(xtraj(:,k), utraj(:,k), xd, Q, R);
        
        % Get dynamics gradients AND Hessians (TRUE DDP requirement)
        [fx_cont, fu_cont, fxx_cont, fux_cont, fuu_cont] = cartpole_grads_and_hessians_full(xtraj(:,k), utraj(:,k), param);
        
        % Discretize dynamics (first-order)
        fx = eye(nX) + dt * fx_cont;
        fu = dt * fu_cont;
        
        % Discretize dynamics Hessians (second-order)
        fxx = dt * fxx_cont;
        fux = dt * fux_cont;
        fuu = dt * fuu_cont;
        
        % Compute Q-function terms (TRUE DDP version with dynamics Hessians)
        [Qx, Qu, Qxx, Qux, Quu] = Q_terms_true_ddp(gx, gu, gxx, gux, guu, fx, fu, fxx, fux, fuu, Vx, Vxx);
        
        % Ensure Quu symmetry before regularization
        Quu = 0.5 * (Quu + Quu');
        
        % CRITICAL: Add regularization with adaptive scaling
        Quu_reg = Quu + mu * eye(nU);
        
        % Check condition number before attempting inversion
        cond_num = cond(Quu_reg);
        if cond_num > 1e12 || any(isnan(Quu_reg(:))) || any(isinf(Quu_reg(:)))
            success = false;
            expected_improvement = 0;
            return;
        end
        
        % Check if Quu_reg is positive definite using Cholesky
        [R_chol, p] = chol(Quu_reg);
        if p ~= 0
            success = false;
            expected_improvement = 0;
            return;
        end
        
        % Compute gains using Cholesky factorization for better numerics
        ktraj(:,k) = -R_chol \ (R_chol' \ Qu);
        Ktraj(:,:,k) = -R_chol \ (R_chol' \ Qux);
        
        % Update expected cost reduction
        expected_improvement = expected_improvement + ktraj(:,k)' * Qu;
        
        % CRITICAL: Update value function using REGULARIZED Quu
        % This is key for numerical stability!
        Vx = Qx + Ktraj(:,:,k)' * Quu_reg * ktraj(:,k) + Ktraj(:,:,k)' * Qu + Qux' * ktraj(:,k);
        Vxx = Qxx + Ktraj(:,:,k)' * Quu_reg * Ktraj(:,:,k) + Ktraj(:,:,k)' * Qux + Qux' * Ktraj(:,:,k);
        
        % CRITICAL: Symmetrize Vxx for numerical stability
        Vxx = 0.5 * (Vxx + Vxx');
        
        % Ensure Vxx remains positive semi-definite
        [V_eig, D_eig] = eig(Vxx);
        D_eig = max(D_eig, 0);  % Set negative eigenvalues to zero
        Vxx = V_eig * D_eig * V_eig';
    end
end

%==========================================================================
%   Complete dynamics Hessians computation
%==========================================================================
function [fx, fu, fxx, fux, fuu] = cartpole_grads_and_hessians_full(x, u, param)
    [fx, fu] = cartpole_grads(0, x, u, param);

    mc = param.mc; mp = param.mp; l = param.l; g = param.g; b = param.b; d = param.d;

    x1 = x(1); x2 = x(2); x3 = x(3); x4 = x(4); %#ok<NASGU>
    s = sin(x2); c = cos(x2);

    nX = 4; nU = 1;
    fxx = zeros(nX, nX, nX);
    fux = zeros(nU, nX, nX);
    fuu = zeros(nU, nU, nX); 

    %：D = mc + mp*s^2
    D   = mc + mp*s^2;
    Dp  = 2*mp*s*c;                % dD/dtheta
    Dpp = 2*mp*(c^2 - s^2);        % d2D/dtheta2

    %-----------------------------
    % f3 = N3 / D
    %-----------------------------
    %   N3 = u - b*x3 + d*x4*c/l + mp*s*(l*x4^2 + g*c)
    % : N3 = u - b*x3 + (d/l)*x4*c + mp*l*x4^2*s + mp*g*s*c
    N3   = u - b*x3 + (d/l)*x4*c + mp*l*x4^2*s + mp*g*s*c;
    N3p  =        -(d/l)*x4*s + mp*l*x4^2*c + mp*g*(c^2 - s^2);       % dN3/dθ
    N3pp =        -(d/l)*x4*c - mp*l*x4^2*s - 4*mp*g*s*c;              % d2N3/dθ2  (note -4*mp*g*s*c)

    N3_x3    = -b;
    N3_x4    = (d/l)*c + 2*mp*l*x4*s;
    N3_px3   = 0;                                  % d^2N3/dθdx3
    N3_px4   = -(d/l)*s + 2*mp*l*x4*c;             % d^2N3/dθdx4
    N3_x4x4  = 2*mp*l*s;                           % d^2N3/dx4^2

    %  f = N(θ,·)/D(θ), 
    %   f_{θθ} = (N''/D) - (N*D''/D^2) - 2*(N'*D'/D^2) + 2*N*(D')^2/D^3
    %   f_{θx} = (N_{θx}*D - N_x*D') / D^2
    %   f_{xx} = N_{xx} / D 

    % fxx(2,2,3) : d^2 f3 / dθ^2
    fxx(2,2,3) =  N3pp/D  - N3*Dpp/(D^2) - 2*N3p*Dp/(D^2) + 2*N3*(Dp^2)/(D^3);

    % fxx(2,3,3) = fxx(3,2,3) : d^2 f3 / dθ dx3
    val = (N3_px3*D - N3_x3*Dp) / (D^2);   % N3_px3=0
    fxx(2,3,3) = val; fxx(3,2,3) = val;

    % fxx(2,4,3) = fxx(4,2,3) : d^2 f3 / dθ dx4
    val = (N3_px4*D - N3_x4*Dp) / (D^2);
    fxx(2,4,3) = val; fxx(4,2,3) = val;

    % fxx(3,3,3) = 0  
    % fxx(3,4,3) = 0 
    % fxx(4,4,3) : d^2 f3 / dx4^2
    fxx(4,4,3) = N3_x4x4 / D;
    %   N4 = -u*c + b*x3*c - d*(mc+mp)*x4/(mp*l) - mp*l*x4^2*c*s - (mc+mp)*g*s/(mp*l)
    N4   = -u*c + b*x3*c - d*(mc+mp)*x4/(mp*l) - mp*l*x4^2*c*s - (mc+mp)*g*s/(mp*l);
    N4p  =  u*s - b*x3*s - mp*l*x4^2*(c^2 - s^2) - (mc+mp)*g*c/(mp*l);          % dN4/dθ
    N4pp =  u*c - b*x3*c + 4*mp*l*x4^2*s*c + (mc+mp)*g*s/(mp*l);                 % d2N4/dθ2

    N4_x3    =  b*c;
    N4_x4    = - d*(mc+mp)/(mp*l) - 2*mp*l*x4*c*s;
    N4_px3   = - b*s;                           % d^2N4/dθdx3
    N4_px4   = - 2*mp*l*x4*(c^2 - s^2);         % d^2N4/dθdx4
    N4_x4x4  = - 2*mp*l*c*s;                    % d^2N4/dx4^2

    %：f4 = N4 / (l*D) = (1/l) * (N4 / D)

    % fxx(2,2,4) : d^2 f4 / dθ^2
    fxx(2,2,4) = ( N4pp/D  - N4*Dpp/(D^2) - 2*N4p*Dp/(D^2) + 2*N4*(Dp^2)/(D^3) ) / l;

    % fxx(2,3,4) = fxx(3,2,4) : d^2 f4 / dθ dx3
    val = (N4_px3*D - N4_x3*Dp) / (D^2) / l;
    fxx(2,3,4) = val; fxx(3,2,4) = val;

    % fxx(2,4,4) = fxx(4,2,4) : d^2 f4 / dθ dx4
    val = (N4_px4*D - N4_x4*Dp) / (D^2) / l;
    fxx(2,4,4) = val; fxx(4,2,4) = val;

    % fxx(3,3,4) = 0 
    % fxx(3,4,4) = 0
    % fxx(4,4,4) : d^2 f4 / dx4^2
    fxx(4,4,4) = (N4_x4x4 / D) / l;    % = -2*mp*c*s / D


    % f3 = N3/D,  N3_u = 1,  N3_{ux} = 0 except x=theta? 
    %   f_{u xj} = ∂/∂xj ( N_u/D ) = (N_{uxj}*D - N_u*D_xj)/D^2 = -(N_u*D_xj)/D^2
    %   D 只依赖 θ
    %     fux(1,2,3) = -(1 * Dp)/D^2,  xj≠θ is 0
    fux(1,1,3) = 0;
    fux(1,2,3) = - Dp / (D^2);
    fux(1,3,3) = 0;
    fux(1,4,3) = 0;

    % f4 = N4/(lD),  N4_u = -c,  N4_{uxj}=0
    %   f_{u xj} = ∂/∂xj ( -c / (lD) ) = -( -s*δ_{j2}*D - c*D_{xj} ) / (lD^2)
    %   ：(s*D + c*Dp)/(l D^2) *? 
    %   对 θ:  ∂θ[ -c/(lD) ] = ( s/(lD) ) + ( c*Dp/(lD^2) )
    %   但这是 f_{θu}，而要 f_{uθ}（等价），所以：
    fux(1,1,4) = 0;
    fux(1,2,4) = ( s*D + c*Dp ) / (l * D^2);
    fux(1,3,4) = 0;
    fux(1,4,4) = 0;

    % 对称化每个 k 的 fxx(i,j,k)
    for k = 1:nX
        fxx(:,:,k) = 0.5 * (fxx(:,:,k) + fxx(:,:,k)');
    end
end

%==========================================================================
%   TRUE DDP Q-terms (includes dynamics second-order terms!)
%==========================================================================
function [Qx, Qu, Qxx, Qux, Quu] = Q_terms_true_ddp(gx, gu, gxx, gux, guu, fx, fu, fxx, fux, fuu, Vx, Vxx)
    nX = length(gx);
    nU = length(gu);
    
    % Standard terms (same as iLQR with second-order cost)
    Qx = gx + fx' * Vx;
    Qu = gu + fu' * Vx;
    
    % Second-order terms WITHOUT dynamics Hessians first
    Qxx = gxx + fx' * Vxx * fx;
    Qux = gux + fu' * Vxx * fx;
    Quu = guu + fu' * Vxx * fu;
    
    % Add the contraction of dynamics Hessians with Vx (TRUE DDP)
    % This is what distinguishes DDP from iLQR!
    for i = 1:nX
        if abs(Vx(i)) > 1e-12  % Skip if Vx(i) is essentially zero
            Qxx = Qxx + Vx(i) * squeeze(fxx(:,:,i));
            Qux = Qux + Vx(i) * squeeze(fux(:,:,i));
            Quu = Quu + Vx(i) * squeeze(fuu(:,:,i));
        end
    end
    
    % Ensure symmetry
    Qxx = 0.5 * (Qxx + Qxx');
    Quu = 0.5 * (Quu + Quu');
end

%==========================================================================
%   DDP forward pass with line search
%==========================================================================
function [xtraj, utraj, J] = forward_pass_ddp(x0, xtraj0, utraj0, ktraj, Ktraj, N, dt, param, Q, R, Qf, xd)
    % Line search parameters
    alphas = [1.0, 0.75, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625];
    
    J_best = inf;
    xtraj_best = xtraj0;
    utraj_best = utraj0;
    
    for alpha = alphas
        x = x0;
        J = 0;
        xtraj_temp = zeros(size(xtraj0));
        utraj_temp = zeros(size(utraj0));
        
        valid = true;
        for k = 1:N-1
            xtraj_temp(:,k) = x;
            
            % Control update with line search parameter
            u = utraj0(:,k) + alpha * ktraj(:,k) + Ktraj(:,:,k) * (x - xtraj0(:,k));
            
            % Bound control input if necessary
            u_max = 100;
            u = max(min(u, u_max), -u_max);
            
            utraj_temp(:,k) = u;
            J = J + stage_cost(x, u, Q, R);
            
            % Integrate dynamics
            xdot = cartpole_dynamics(0, x, u, param);
            x = x + dt * xdot;
            
            % Check for numerical issues
            if any(isnan(x)) || any(isinf(x))
                valid = false;
                break;
            end
        end
        
        if valid
            xtraj_temp(:,N) = x;
            J = J + terminal_cost(x, xd, Qf);
            
            % Keep best trajectory
            if J < J_best
                J_best = J;
                xtraj_best = xtraj_temp;
                utraj_best = utraj_temp;
            end
        end
    end
    
    xtraj = xtraj_best;
    utraj = utraj_best;
    J = J_best;
end

%==========================================================================
%   DDP cost gradients
%==========================================================================
function [gx, gu, gxx, gux, guu] = cost_gradients_ddp(x, u, xd, Q, R)
    % First-order gradients
    gx = Q * x;  % For regulation around origin
    gu = R * u;
    
    % Second-order gradients
    gxx = Q;
    gux = zeros(size(u,1), length(x));
    guu = R;
end

%==========================================================================
%   Stage cost function
%==========================================================================
function J = stage_cost(x, u, Q, R)
    J = 0.5 * x' * Q * x + 0.5 * u' * R * u;
end

%==========================================================================
%   Terminal cost function
%==========================================================================
function Jf = terminal_cost(x, xd, Qf)
    Jf = 0.5 * (x - xd)' * Qf * (x - xd);
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
%   cart pole gradients (Jacobians)
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