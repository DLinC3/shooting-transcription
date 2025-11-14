%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% direct_transcription_fmincon - Optimal swing-up trajectory via direct transcription
%
%   - init_cartpole_params()    : Initialize system parameters
%   - cartpole_dynamics(t,x,u,param) : System dynamics (returns xdot)
%   - draw_cartpole(t,x,param)  : Visualization function
%


function direct_transcription_fmincon

clear all; close all;

%% --- Initialize System and Parameters ---
fprintf('Initializing cartpole swing-up optimization...\n');

% Add path to dynamics functions
addpath("dynamical_systems/");

% Initialize system parameters
param = init_cartpole_params();

% Define function handles
DYNAMICS = @cartpole_dynamics;
DRAW = @draw_cartpole;

%% --- Discretization Settings ---
N = 600;          % Number of time steps (knot points)
param.dt = 1/120;  % Time step size [seconds]
% Total time: T = N * dt = 5 seconds

%% --- Configure Optimization Parameters ---
param = opt_param(param);
param.nX = 4;     % State dimension [position, angle, velocity, angular_velocity]
param.nU = 1;     % Control dimension [cart force]
param.xd = [0, pi]; % Desired final position for visualization
param.plot = 0;   % Disable plotting during optimization (set to 1 to enable)

%% --- Solve Optimization Problem ---
fprintf('Starting optimization with FMINCON...\n');
fprintf('Problem size: %d states × %d time steps = %d state variables\n', param.nX, N, param.nX*N);
fprintf('              %d control × %d time steps = %d control variables\n', param.nU, N, param.nU*N);

tic;
[w, F, info] = transcription_fmincon(N, DYNAMICS, param);
elapsed_time = toc;

fprintf('Optimization completed in %.2f seconds\n', elapsed_time);
fprintf('Exit flag: %d\n', info);

%% --- Extract Solution Trajectories ---
% Unpack the optimization solution vector w = [h; X(:); U(:)]
xtape = reshape(w(2:(1+param.nX*N)), param.nX, N);    % State trajectory [4 × N]
utape = reshape(w(param.nX*N+2:end), param.nU, N);    % Control trajectory [1 × N]

% Display final state
fprintf('\nFinal state achieved:\n');
fprintf('  Position:         %.4f m (target: 0)\n', xtape(1,end));
fprintf('  Angle:            %.4f rad (target: π = %.4f)\n', xtape(2,end), pi);
fprintf('  Velocity:         %.4f m/s (target: 0)\n', xtape(3,end));
fprintf('  Angular velocity: %.4f rad/s (target: 0)\n', xtape(4,end));

%% --- Visualization: Plot Trajectories ---

% Figure 1: Phase portrait - Cart position vs Pole angle
figure(2); clf;
plot(xtape(1,:), xtape(2,:), 'b-', 'LineWidth', 1.5);
hold on;
plot(0, 0, 'go', 'MarkerSize', 8, 'MarkerFaceColor', 'g');  % Start point
plot(0, pi, 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r'); % Target point
xlabel('Cart Position [m]'); 
ylabel('Pole Angle [rad]');
title('Phase Portrait: Cart Position vs Pole Angle');
grid on; 
legend('Trajectory', 'Start', 'Target', 'Location', 'best');
hold off;

% Figure 2: Cart position over time
figure(3); clf;
time_vec = (0:N-1) * param.dt;
plot(time_vec, xtape(1,:), 'b-', 'LineWidth', 1.5);
hold on;
plot(0, 0, 'go', 'MarkerSize', 8, 'MarkerFaceColor', 'g');
xlabel('Time [s]'); 
ylabel('Cart Position [m]');
title('Cart Position Trajectory');
grid on;
hold off;

% Figure 3: Pole angle over time
figure(4); clf;
plot(time_vec, xtape(2,:), 'b-', 'LineWidth', 1.5);
hold on;
plot(0, 0, 'go', 'MarkerSize', 8, 'MarkerFaceColor', 'g');
yline(pi, 'r--', 'Target', 'LineWidth', 1.5);
xlabel('Time [s]'); 
ylabel('Pole Angle [rad]');
title('Pole Angle Trajectory');
grid on;
hold off;

% Figure 4: Control input over time
figure(5); clf;
plot(time_vec, utape, 'r-', 'LineWidth', 1.5);
xlabel('Time [s]'); 
ylabel('Control Force [N]');
title('Control Input Trajectory');
grid on;
ylim([param.opt.umin*1.1, param.opt.umax*1.1]);

% Figure 5: Phase portrait - Pole angle vs Angular velocity
figure(6); clf;
plot(xtape(2,:), xtape(4,:), 'b-', 'LineWidth', 1.5);
xlabel('Pole Angle [rad]'); 
ylabel('Angular Velocity [rad/s]');
title('Phase Portrait: Pole Angle vs Angular Velocity');
axis tight;
grid on;

% Figure 6: Phase portrait - Cart position vs Pole angle (clean version)
figure(7); clf;
plot(xtape(1,:), xtape(2,:), 'b-', 'LineWidth', 0.8);
axis tight;
grid off;
set(gca, 'Box', 'off');
xlabel('Cart Position [m]'); 
ylabel('Pole Angle [rad]');
title('Clean Phase Portrait');

%% --- Animation of Solution ---
fprintf('\nAnimating solution...\n');
figure(25); clf;
v = VideoWriter('fmincon_ani.mp4', 'MPEG-4');
v.FrameRate = 120;  % 
v.Quality = 100;     % (0-100)
open(v);
for i = 1:size(xtape, 2)
    DRAW((i-1)*param.dt, xtape(:,i), param);
    frame = getframe(gcf);
    writeVideo(v, frame);
    drawnow;
    % pause(0.01);  % Small pause for smoother animation
end
close(v);

fprintf('Animation complete.\n');

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% OPT_PARAM - Configure optimization parameters and constraints
%
% This function sets up all bounds, constraints, and cost function weights
% for the direct transcription optimization problem.
%
% Input:
%   param - Base parameter structure
% Output:
%   param - Updated structure with optimization settings
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function param = opt_param(param)

    %% --- State Configuration ---
    % Initial state: cart at origin, pole hanging down
    param.opt.x0 = [0; 0; 0; 0];  % [position; angle; velocity; angular_velocity]
    
    % Target state: cart at origin, pole inverted (up)
    param.opt.xd = [0; pi; 0; 0]; % [position; angle; velocity; angular_velocity]
    
    %% --- Terminal State Constraints ---
    % Define a small box around the desired final state
    terminal_tolerance = 5e-4;  % Very tight tolerance for accurate terminal state
    param.opt.xfmax = [terminal_tolerance; 
                       pi + 0.01*terminal_tolerance; 
                       terminal_tolerance; 
                       terminal_tolerance];
    param.opt.xfmin = [-terminal_tolerance; 
                       pi - 0.01*terminal_tolerance; 
                       -terminal_tolerance; 
                       -terminal_tolerance];
    
    %% --- State Bounds ---
    % Physical limits on states during trajectory
    param.opt.xmax = [10; 10; 10; 10];   % Upper bounds for all states
    param.opt.xmin = [-10; -10; -10; -10]; % Lower bounds for all states
    
    %% --- Control Bounds ---
    % Limits on cart force
    param.opt.umax = 50;   % Maximum force [N]
    param.opt.umin = -50;  % Minimum force [N]
    
    % Legacy parameters (kept for compatibility)
    param.opt.umax1 = 50;
    param.opt.umin1 = -50;
    
    %% --- System Dimensions ---
    param.opt.nX = 4;  % Number of states
    param.opt.nU = 1;  % Number of controls
    
    %% --- Cost Function Weights ---
    % Running cost: L = x'*Q*x + u'*R*u
    param.opt.Q = diag([1,    % Position weight
                        1,    % Angle weight  
                        0.1,  % Velocity weight
                        0.1]); % Angular velocity weight
    
    % Terminal cost: Lf = (x_N - x_d)'*Qf*(x_N - x_d)
    param.opt.Qf = diag([10,   % Position weight (moderate)
                         500,  % Angle weight (very high - critical)
                         1,    % Velocity weight
                         1]);  % Angular velocity weight
    
    % Control effort weight
    param.opt.R = 0.01;  % Small weight to allow aggressive control
    
    %% --- Time Step Bounds ---
    % Fixed time step (no time optimization)
    param.opt.hmin = param.dt;
    param.opt.hmax = param.dt;

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% transcription_fmincon - Direct transcription optimization using FMINCON
%
% This function formulates and solves the direct transcription problem
% using MATLAB's FMINCON nonlinear programming solver.
%
% Inputs:
%   N        - Number of time steps
%   DYNAMICS - Function handle for system dynamics
%   param    - Parameter structure
% Outputs:
%   w    - Optimal solution vector [h; X(:); U(:)]
%   F    - Final cost value
%   info - FMINCON exit flag
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [w, F, info] = transcription_fmincon(N, DYNAMICS, param)

    % Store initial state for constraint enforcement
    param.opt.xi = param.opt.x0;
    
    nU = param.opt.nU;
    nX = param.opt.nX;
    
    %% --- Initialize Control Trajectory ---
    if isfield(param.opt, 'u0s')
        u0 = param.opt.u0s;  % Use provided initial guess
    else
        u0 = zeros(nU*N, 1); % Zero initialization
    end
    
    % Control bounds for all time steps
    ulow = param.opt.umin * ones(nU*N, 1);
    uhigh = param.opt.umax * ones(nU*N, 1);
    
    %% --- Initialize State Trajectory ---
    x0 = [];
    xlow = [];
    xhigh = [];
    
    % Build state vectors with linear interpolation as initial guess
    for k = 1:nX
        % Initial guess: linear interpolation from x0 to xd
        x0 = [x0; linspace(param.opt.x0(k), param.opt.xd(k), N)];
        
        % State bounds for all time steps
        xlow = [xlow; param.opt.xmin(k) * ones(1, N)];
        xhigh = [xhigh; param.opt.xmax(k) * ones(1, N)];
    end
    
    % Override with provided initial guess if available
    if isfield(param.opt, 'x0s')
        x0 = param.opt.x0s;
    end
    
    %% --- Enforce Boundary Conditions ---
    % Fix initial state (equality constraint via bounds)
    xlow(:, 1) = param.opt.x0;
    xhigh(:, 1) = param.opt.x0;
    
    % Constrain terminal state to target box
    xlow(:, end) = param.opt.xfmin;
    xhigh(:, end) = param.opt.xfmax;
    
    %% --- Time Step Bounds ---
    hhigh = param.opt.hmax;
    hlow = param.opt.hmin;
    
    %% --- Pack Decision Variables ---
    % Decision vector: w = [h; X(:); U(:)]
    % where h is time step, X is states, U is controls
    w0 = [5/N;        % Initial time step guess (will be overridden by bounds)
          x0(:);      % Vectorized state trajectory
          u0];        % Vectorized control trajectory
    
    wlow = [hlow;     % Lower bound on time step
            xlow(:);  % Lower bounds on states
            ulow];    % Lower bounds on controls
    
    whigh = [hhigh;   % Upper bound on time step
             xhigh(:); % Upper bounds on states
             uhigh];   % Upper bounds on controls
    
    %% --- Configure FMINCON Options ---
    options = optimset('Algorithm', 'interior-point', ...  % Better for large problems
                      'Display', 'iter', ...               % Show iterations
                      'MaxFunEvals', 300000, ...           % Maximum function evaluations
                      'TolFun', 1e-4, ...                  % Function tolerance
                      'TolX', 1e-4, ...                     % Variable tolerance
                      'TolCon', 1e-4, ...                   % Constraint tolerance
                      'MaxIter', 10000);                     % Maximum iterations
    
    %% --- Solve Optimization Problem ---
    fprintf('Decision variables: %d\n', length(w0));
    fprintf('Equality constraints: %d\n', (N-1)*nX);
    
    [w, F, info] = fmincon(@(w) cost_function(w, N, param), ... % Objective function
                          w0, ...                                % Initial guess
                          [], [], ...                            % No linear inequality constraints
                          [], [], ...                            % No linear equality constraints
                          wlow, whigh, ...                       % Variable bounds
                          @(w) constraints(w, N, DYNAMICS, param), ... % Nonlinear constraints
                          options);

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% COST_FUNCTION - Evaluate the objective function for optimization
%
% Computes the total cost J = integral(L) + Lf
% where L is the running cost and Lf is the terminal cost
%
% Input:
%   w     - Decision vector [h; X(:); U(:)]
%   N     - Number of time steps
%   param - Parameter structure
% Output:
%   f - Objective function value
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function f = cost_function(w, N, param)
    % Persistent counter for tracking function evaluations (optional)
    persistent k;
    if isempty(k)
        k = 1;
    end
    
    nX = param.opt.nX;
    nU = param.opt.nU;
    
    %% --- Unpack Decision Variables ---
    h = w(1);                                    % Time step
    X = reshape(w(2:(1+nX*N)), nX, N);         % States [nX × N]
    u = reshape(w(nX*N+2:end), nU, N);         % Controls [nU × N]
    
    %% --- Compute Total Cost ---
    J = 0;  % Initialize total cost
    
    % Accumulate running cost over trajectory
    for i = 2:N
        t = 0;  % Time is not used in cost function
        
        % Running cost: L = x'*Q*x + u'*R*u
        % Integrated using rectangle rule: J += L * dt
        J = J + cost(t, X(:,i-1), u(:,i-1), param) * h;
    end
    
    % Add terminal cost: Lf = (x_N - x_d)'*Qf*(x_N - x_d)
    terminal_error = X(:,N) - param.opt.xd;
    J = J + final_cost(0, terminal_error, [], param);
    
    % Return the total cost
    f = J;
    
    %% --- Optional: Visualization During Optimization ---
    if param.plot == 1
        % Plot current trajectory (useful for debugging)
        figure(100); clf;
        plot(X(1,:), X(2,:), 'b-', 'LineWidth', 1.5);
        hold on;
        plot(0, 0, 'ko', 'MarkerSize', 5, 'MarkerFaceColor', [0 0 0]);
        xlabel('X (m)', 'FontSize', 18);
        ylabel('Theta (rad)', 'FontSize', 18);
        h_ax = gca;
        set(h_ax, 'FontSize', 16);
        axis equal; 
        axis([-4 1 -1 1.75]);
        title(sprintf('Iteration %d, Cost = %.3f', k, J));
        drawnow;
        hold off;
    end
    
    k = k + 1;  % Increment function evaluation counter
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CONSTRAINTS - Evaluate nonlinear equality constraints
%
% Enforces dynamics constraints using Euler integration:
% x_{k+1} = x_k + f(x_k, u_k) * dt
%
% Inputs:
%   w        - Decision vector [h; X(:); U(:)]
%   N        - Number of time steps
%   DYNAMICS - System dynamics function
%   param    - Parameter structure
% Outputs:
%   c   - Inequality constraints (empty for this problem)
%   ceq - Equality constraints (dynamics residuals)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [c, ceq] = constraints(w, N, DYNAMICS, param)
    
    nX = param.opt.nX;
    nU = param.opt.nU;
    
    %% --- Unpack Decision Variables ---
    h = w(1);                                    % Time step
    X = reshape(w(2:(1+nX*N)), nX, N);         % States [nX × N]
    u = reshape(w(nX*N+2:end), nU, N);         % Controls [nU × N]
    
    %% --- Initialize Constraint Vectors ---
    c = [];   % No inequality constraints
    ceq = []; % Equality constraints (dynamics)
    
    %% --- Enforce Dynamics Constraints ---
    % For each time step, enforce Euler integration
    for i = 2:N
        t = 0;  % Time argument (not used in autonomous system)
        
        % Compute state derivative at previous time step
        xdot = DYNAMICS(t, X(:,i-1), u(:,i-1), param);
        
        % Dynamics residual (should equal zero):
        % r_i = x_i - (x_{i-1} + f(x_{i-1}, u_{i-1}) * h)
        f_dyn = X(:,i) - (X(:,i-1) + xdot * h);
        
        % Append to equality constraint vector
        ceq = [ceq; f_dyn];
    end
    
    % Total equality constraints: (N-1) * nX
    % Each represents dynamics constraint at one time step
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FINAL_COST - Compute terminal cost
%
% Terminal cost penalizes deviation from target state:
% Lf = (x_N - x_d)' * Qf * (x_N - x_d)
%
% Inputs:
%   t     - Time (unused)
%   x     - Terminal state error (x_N - x_d)
%   u     - Control (unused for terminal cost)
%   param - Parameter structure
% Output:
%   J - Terminal cost value
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function J = final_cost(t, x, u, param)
    Qf = param.opt.Qf;  % Terminal cost weight matrix
    J = x' * Qf * x;    % Quadratic form
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% COST - Compute running cost
%
% Running cost penalizes state deviation and control effort:
% L = x' * Q * x + u' * R * u
%
% Inputs:
%   t     - Time (unused)
%   x     - Current state
%   u     - Current control
%   param - Parameter structure
% Output:
%   J - Running cost value
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function J = cost(t, x, u, param)
    Q = param.opt.Q;  % State cost weight matrix
    R = param.opt.R;  % Control cost weight
    
    J = x' * Q * x + u' * R * u;  % LQR-style quadratic cost
end