function direct_transcription_SNOPT
% direct_transcription_SNOPT
% Direct transcription optimization for cartpole swing-up using analytical gradients and SNOPT.
%
% This implementation uses Euler integration for dynamics discretization
% and provides exact gradients to SNOPT for efficient optimization.
%
% External dependencies (must be on MATLAB path):
%   - init_cartpole_params()    : Initialize system parameters
%   - cartpole_dynamics(t,x,u,param) : System dynamics (returns xdot)
%   - draw_cartpole(t,x,param)  : Visualization function
%   - cartpole_grads(t,x,u,param) : Dynamics gradients (returns [dfdx, dfdu])
%
% Tested with SNOPT7 MATLAB interface.
%
% Author: Denglin
% Date: 05/13/2025

clear all; close all;

%% --- User Paths Configuration ---
% Edit these paths to match your local setup
addpath("dynamical_systems/");                     % Directory containing dynamics functions
addpath(genpath('C:\Users\dcheng32\snopt7_matlab'));  % SNOPT MATLAB interface
setenv('SNOPT_LICENSE', 'C:\Users\dcheng32\snopt7_matlab\snopt7.lic');

%% --- Problem Setup ---
% Initialize system parameters
param = init_cartpole_params();

% Define function handles for dynamics and visualization
DYNAMICS  = @cartpole_dynamics;
DRAW      = @draw_cartpole;
GRADIENTS = @cartpole_grads;

% Discretization parameters
N        = 600;       % Number of time nodes (knot points)
param.dt = 1/120;      % Fixed time step size [seconds]

% Configure optimization parameters
param    = opt_param(param);
param.nX = 4;         % State dimension (x, theta, xdot, thetadot)
param.nU = 1;         % Control dimension (cart force)
param.xd = [0, pi];   % Desired state for plotting only
param.plot = 0;       % Plotting flag during optimization

%% --- Solve Optimization Problem with SNOPT ---
fprintf('Starting optimization with SNOPT...\n');
tic;
[w, F, info] = transcription_SNOPT(N, DYNAMICS, GRADIENTS, param);
elapsed_time = toc;
fprintf('Optimization completed in %.2f seconds\n', elapsed_time);
fprintf('SNOPT exit code: %d\n', info);

%% --- Extract and Visualize Results ---
nX = param.nX; 
nU = param.nU;

% Unpack state and control trajectories from solution vector
xtape = reshape(w(2:(1+nX*N)), nX, N);    % State trajectory [nX x N]
utape = reshape(w(nX*N+2:end), nU, N);    % Control trajectory [nU x N]

% Display final state
fprintf('Final state: [%.3f, %.3f, %.3f, %.3f]\n', xtape(:,end));

% Plot 1: Cart position vs pole angle (x1-x2 phase portrait)
figure(2); clf; hold on;
plot(xtape(1,:), xtape(2,:), 'b-', 'LineWidth', 1.5);
plot(0, 0, 'go', 'MarkerSize', 8, 'MarkerFaceColor', 'g');  % Initial position
plot(0, pi, 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r'); % Target position
xlabel('Cart Position [m]'); ylabel('Pole Angle [rad]');
title('Phase Portrait: Cart Position vs Pole Angle');
grid on; hold off;

% Plot 2: Cart position over time
figure(3); clf;
plot(0:N-1, xtape(1,:), 'b-', 'LineWidth', 1.5);
xlabel('Time Step'); ylabel('Cart Position [m]');
title('Cart Position Trajectory');
grid on;

% Plot 3: Pole angle over time
figure(4); clf;
plot(0:N-1, xtape(2,:), 'b-', 'LineWidth', 1.5);
xlabel('Time Step'); ylabel('Pole Angle [rad]');
title('Pole Angle Trajectory');
grid on;

% Plot 4: Control input over time
figure(5); clf;
plot(0:N-1, utape, 'r-', 'LineWidth', 1.5);
xlabel('Time Step'); ylabel('Control Force [N]');
title('Control Input Trajectory');
grid on;

% Plot 5: Phase portrait - pole angle vs angular velocity
figure(6); clf;
plot(xtape(2,:), xtape(4,:), 'b-', 'LineWidth', 1.5);
xlabel('Pole Angle [rad]'); ylabel('Angular Velocity [rad/s]');
title('Phase Portrait: Angle vs Angular Velocity');
axis tight; grid on;

% Animate the solution
fprintf('Animating solution...\n');
figure(25); clf;
% video
v = VideoWriter('SNOPT_ani.mp4', 'MPEG-4');
v.FrameRate = 120;  % frame
v.Quality = 100;     % Quality (0-100)
open(v);
for i = 1:size(xtape, 2)
    DRAW((i-1)*param.dt, xtape(:,i), param);
    frame = getframe(gcf);
    writeVideo(v, frame);
    drawnow;
end
close(v);
fprintf('Video saved as cartpole_animation.mp4\n');
end % ===== End of Main Function =====


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% OPT_PARAM - Configure optimization parameters and constraints
%
% Input:
%   param - Base parameter structure
% Output:
%   param - Updated with optimization settings
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function param = opt_param(param)
    % Initial and target states
    param.opt.x0  = [0; 0; 0; 0];      % Initial state: cart at origin, pole down
    param.opt.xd  = [0; pi; 0; 0];     % Target state: cart at origin, pole up
    
    % Terminal state constraints (tight box around desired state)
    terminal_tol = 5e-6;  % Tolerance for terminal constraints
    param.opt.xfmax = [ terminal_tol; pi + 0.01*terminal_tol;  terminal_tol;  terminal_tol];
    param.opt.xfmin = [-terminal_tol; pi - 0.01*terminal_tol; -terminal_tol; -terminal_tol];
    % State bounds (safety limits)
    param.opt.xmax = [10; 10; 10; 10];  % [position, angle, velocity, angular_velocity]
    param.opt.xmin = -param.opt.xmax;
    
    % Control bounds
    param.opt.umax = 100;   % Maximum force [N]
    param.opt.umin = -100;  % Minimum force [N]
    
    % System dimensions
    param.opt.nX = 4;  % State dimension
    param.opt.nU = 1;  % Control dimension
    
    % Cost function weights (LQR-style)
    param.opt.Q  = diag([0.01, 0.01, 0.01, 0.01]);  % Running cost state weights
    param.opt.Qf = diag([1, 50, 1, 1]);              % Terminal cost weights (heavy on angle)
    param.opt.R  = 0.01;                             % Control effort weight
    
    % Time step bounds (fixed for this problem)
    param.opt.hmin = param.dt;
    param.opt.hmax = param.dt;
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% transcription_SNOPT - Main optimization routine using direct transcription
%
% Inputs:
%   N         - Number of time nodes
%   DYNAMICS  - Function handle for system dynamics
%   GRADIENTS - Function handle for dynamics gradients
%   param     - Parameter structure
% Outputs:
%   w    - Optimal solution vector [h; X(:); U(:)]
%   F    - Final constraint values
%   info - SNOPT exit status
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [w, F, info] = transcription_SNOPT(N, DYNAMICS, GRADIENTS, param)
    
    nX = param.opt.nX;
    nU = param.opt.nU;
    
    %% --- Initialize Decision Variables ---
    
    % Control trajectory initial guess (zero or provided)
    if isfield(param.opt, 'u0s')
        u0 = param.opt.u0s;
    else
        u0 = zeros(nU*N, 1);
    end
    ulow  = param.opt.umin * ones(nU*N, 1);
    uhigh = param.opt.umax * ones(nU*N, 1);
    
    % State trajectory initial guess (linear interpolation or provided)
    x0    = [];
    xlow  = [];
    xhigh = [];
    for k = 1:nX
        x0    = [x0; linspace(param.opt.x0(k), param.opt.xd(k), N)];
        xlow  = [xlow; param.opt.xmin(k) * ones(1, N)];
        xhigh = [xhigh; param.opt.xmax(k) * ones(1, N)];
    end
    if isfield(param.opt, 'x0s')
        x0 = param.opt.x0s;
    end
    
    % Enforce initial and terminal state constraints via bounds
    xlow(:, 1)   = param.opt.x0;       % Fix initial state
    xhigh(:, 1)  = param.opt.x0;
    xlow(:, end) = param.opt.xfmin;    % Terminal state box constraint
    xhigh(:, end) = param.opt.xfmax;
    
    % Time step bounds
    hlow  = param.opt.hmin;
    hhigh = param.opt.hmax;
    
    % Pack decision vector: w = [h; X(:); U(:)]
    w0    = [2/N; x0(:); u0];  % Initial h value doesn't matter (it's bounded)
    wlow  = [hlow; xlow(:); ulow];
    whigh = [hhigh; xhigh(:); uhigh];
    
    %% --- Define Constraint Structure ---
    
    % Total rows: 1 objective + (N-1)*nX equality constraints for dynamics
    nF = 1 + (N-1)*nX;
    Flow  = [-inf; zeros((N-1)*nX, 1)];  % Objective unbounded below, dynamics = 0
    Fhigh = [ inf; zeros((N-1)*nX, 1)];  % Objective unbounded above, dynamics = 0
    
    % Helper functions for indexing decision vector
    idx_h = 1;                                                    % Time step index
    idxX = @(k) (2 + (k-1)*nX) : (1 + k*nX);                    % State block k
    idxU = @(k) (2 + nX*N + (k-1)*nU) : (1 + nX*N + k*nU);      % Control block k
    
    %% --- Build Jacobian Sparsity Pattern ---
    % Define which entries of the Jacobian are non-zero
    
    iGfun = [];  % Row indices of non-zero Jacobian entries
    jGvar = [];  % Column indices of non-zero Jacobian entries
    
    % (A) Objective gradient: depends on all variables
    num_vars = 1 + nX*N + nU*N;  % Total number of variables
    iGfun = [iGfun; ones(num_vars, 1)];  % All in row 1 (objective)
    jGvar = [jGvar; (1:num_vars)'];      % All columns
    
    % (B) Dynamics constraints: Euler integration residuals
    % For each time step k = 1..N-1, we have nX constraint equations
    for k = 1:(N-1)
        rowBlock = 1 + (k-1)*nX + (1:nX);  % Rows for dynamics residual at step k
        
        % Dependence on x_{k+1}: Identity matrix (+I)
        iGfun = [iGfun; rowBlock'];
        jGvar = [jGvar; idxX(k+1)'];
        
        % Dependence on x_k: (-I - h*df/dx)
        % Each constraint depends on all state components
        iGfun = [iGfun; repelem(rowBlock', nX)];    % Repeat each row nX times
        jGvar = [jGvar; repmat(idxX(k)', nX, 1)];   % Cycle through columns
        
        % Dependence on u_k: (-h*df/du)
        iGfun = [iGfun; repelem(rowBlock', nU)];
        jGvar = [jGvar; repmat(idxU(k)', nX, 1)];
        
        % Dependence on h: (-f)
        iGfun = [iGfun; rowBlock'];
        jGvar = [jGvar; repmat(idx_h, nX, 1)];
    end
    
    %% --- Configure SNOPT Options ---
    setSNOPTParam('Major Iterations Limit', 10000);
    setSNOPTParam('Minor Iterations Limit', 500);
    setSNOPTParam('Major Optimality Tolerance', 1e-6);
    setSNOPTParam('Major Feasibility Tolerance', 1e-6);
    setSNOPTParam('Minor Feasibility Tolerance', 1e-6);
    setSNOPTParam('Superbasics Limit', max(2000, 1 + nX*N + nU*N));
    setSNOPTParam('Derivative Option', 1);  % User provides derivatives
    setSNOPTParam('Verify Level', 0);  % Set to 1-3 for debugging
    setSNOPTParam('Iterations Limit', 10000);
    
    %% --- Initialize SNOPT Variables ---
    wmul   = zeros(length(w0), 1);   % Lagrange multipliers for bounds
    wstate = zeros(length(w0), 1);   % State of variables (basic, nonbasic, etc.)
    Fmul   = zeros(length(Flow), 1); % Lagrange multipliers for constraints
    Fstate = zeros(length(Flow), 1); % State of constraints
    
    % Create local wrapper for user function
    function [Fout, Gout] = USERFUN_local(w)
        [Fout, Gout] = userFun(w, N, DYNAMICS, GRADIENTS, param);
    end
    
    % SNOPT options structure
    options.name   = 'Cartpole-DirTran';
    options.start  = 'Cold';  % Cold start (no initial basis)
    options.screen = 'on';    % Display iterations
    
    % Handle parallel execution (if applicable)
    try
        tid = get(getCurrentTask(), 'ID');
    catch
        tid = 0;
    end
    options.printfile = sprintf('cartpole_dirtran_snopt_%d.out', tid);
    
    % Define sparse matrix structures
    A_struct = struct('row', [], 'col', [], 'val', []);  % No linear constraints
    G_struct = struct('row', iGfun, 'col', jGvar);       % Jacobian pattern
    
    ObjAdd = 0;   % Constant added to objective
    ObjRow = 1;   % Row containing objective function
    
    % Verify Jacobian dimensions
    nnz_obj = 1 + param.opt.nX*N + param.opt.nU*N;  % Non-zeros from objective
    nnz_con = (N-1) * (param.opt.nX + param.opt.nX*param.opt.nX + ...
                       param.opt.nX*param.opt.nU + param.opt.nX);  % From constraints
    expected_nnz = nnz_obj + nnz_con;
    assert(length(iGfun) == expected_nnz && length(jGvar) == length(iGfun), ...
           'Jacobian sparsity pattern dimension mismatch');
    
    %% --- Test User Function Before Calling SNOPT ---
    try
        [Ftest, Gtest] = USERFUN_local(w0);
        fprintf('Function evaluation test passed:\n');
        fprintf('  F length = %d (expected = %d)\n', length(Ftest), nF);
        fprintf('  G length = %d (expected = %d)\n', length(Gtest), expected_nnz);
    catch ME
        fprintf('\n>>> ERROR in USERFUN_local(w0):\n%s\n', getReport(ME, 'extended'));
        fprintf('Fix this error before calling SNOPT.\n');
        return;
    end
    
    %% --- Call SNOPT Solver ---
    [w, F, info] = snopt(w0, wlow, whigh, wmul, wstate, ...
                         Flow, Fhigh, Fmul, Fstate, ...
                         @USERFUN_local, ObjAdd, ObjRow, ...
                         A_struct, G_struct, options);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% USERFUN - Compute objective, constraints, and gradients for SNOPT
%
% Inputs:
%   w         - Decision vector [h; X(:); U(:)]
%   N         - Number of time nodes
%   DYNAMICS  - System dynamics function
%   GRADIENTS - Dynamics gradients function
%   param     - Parameters
% Outputs:
%   f - Function values [objective; constraints]
%   G - Gradient values (non-zero entries matching sparsity pattern)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [f, G] = userFun(w, N, DYNAMICS, GRADIENTS, param)
    
    nX = param.opt.nX;
    nU = param.opt.nU;
    
    % Unpack decision variables
    h = w(1);                                    % Time step
    X = reshape(w(2:(1+nX*N)), nX, N);         % States [nX x N]
    U = reshape(w(nX*N+2:end), nU, N);         % Controls [nU x N]
    
    % Initialize containers
    f_dyn = zeros(nX, N-1);      % Dynamics residuals
    J = 0;                       % Total cost
    Jsum_integrand = 0;          % Sum of stage costs (for dJ/dh)
    dJdx = zeros(nX, N);         % Cost gradient w.r.t. states
    dJdu = zeros(nU, N);         % Cost gradient w.r.t. controls
    
    %% --- Compute Stage Costs and Dynamics Residuals ---
    for i = 2:N
        xi = X(:, i-1);
        ui = U(:, i-1);
        
        % Dynamics residual: r_{i-1} = x_i - (x_{i-1} + h*f(x_{i-1}, u_{i-1}))
        xdot = DYNAMICS(0, xi, ui, param);
        f_dyn(:, i-1) = X(:, i) - (xi + h * xdot);
        
        % Stage cost contribution
        Ji = cost(0, xi, ui, param);
        J = J + Ji * h;  % Integrate using rectangle rule
        Jsum_integrand = Jsum_integrand + Ji;
        
        % Stage cost gradients (convert from row to column vectors)
        [dJdxi_row, dJdui_row] = cost_gradients(0, xi, ui, param);
        dJdx(:, i-1) = dJdxi_row.' * h;  % Gradient w.r.t. x_{i-1}
        dJdu(:, i-1) = dJdui_row.' * h;  % Gradient w.r.t. u_{i-1}
    end
    
    %% --- Add Terminal Cost ---
    xN_error = X(:, N) - param.opt.xd;  % Terminal state error
    J = J + final_cost(0, xN_error, [], param);
    
    % Terminal cost gradients
    [dJdxN_row, ~] = final_cost_gradients(0, xN_error, [], param);
    dJdx(:, N) = dJdxN_row.';  % Gradient w.r.t. final state
    dJdu(:, N) = zeros(nU, 1); % No control at final time
    
    % Gradient of cost w.r.t. time step
    dJdh = Jsum_integrand;
    
    % Pack function values: [objective; constraints]
    f = [J; f_dyn(:)];
    
    %% --- Assemble Gradient Values (Matching Sparsity Pattern) ---
    values = [];
    
    % (A) Objective gradients in order: [dJ/dh, dJ/dX(:), dJ/dU(:)]
    values = [values; dJdh];
    values = [values; dJdx(:)];
    values = [values; dJdu(:)];
    
    % (B) Constraint gradients for each time step
    for k = 1:(N-1)
        xi = X(:, k);
        ui = U(:, k);
        xdot = DYNAMICS(0, xi, ui, param);
        [dfdx, dfdu] = GRADIENTS(0, xi, ui, param);
        
        % Gradient w.r.t. x_{k+1}: Identity matrix
        values = [values; ones(nX, 1)];
        
        % Gradient w.r.t. x_k: (-I - h*df/dx)
        % Note: Must transpose for row-major ordering
        block = -eye(nX) - h*dfdx;
        block_transposed = block.';
        values = [values; block_transposed(:)];
        
        % Gradient w.r.t. u_k: (-h*df/du)
        dfdu_scaled = -h * dfdu;
        dfdu_transposed = dfdu_scaled.';
        values = [values; dfdu_transposed(:)];
        
        % Gradient w.r.t. h: (-f)
        values = [values; -xdot];
    end
    
    G = values;
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FINAL_COST_GRADIENTS - Compute terminal cost gradients
% Terminal cost: J_f = (x_N - x_d)' * Q_f * (x_N - x_d)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [dJdx, dJdu] = final_cost_gradients(~, x, ~, param)
    Qf = param.opt.Qf;
    dJdx = 2 * x.' * Qf;  % Row vector: 2*Qf*(x_N - x_d)
    dJdu = 0;             % No control at terminal time
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FINAL_COST - Compute terminal cost value
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function J = final_cost(~, x, ~, param)
    Qf = param.opt.Qf;
    J = x.' * Qf * x;  % Quadratic terminal cost
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% COST_GRADIENTS - Compute running cost gradients
% Running cost: L = x'*Q*x + u'*R*u
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [dJdx, dJdu] = cost_gradients(~, x, u, param)
    Q = param.opt.Q;
    R = param.opt.R;
    dJdx = 2 * x.' * Q;  % Row vector: 2*Q*x
    dJdu = 2 * u.' * R;  % Row vector: 2*R*u
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% COST - Compute running cost value
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function J = cost(~, x, u, param)
    Q = param.opt.Q;
    R = param.opt.R;
    J = x.' * Q * x + u.' * R * u;  % Quadratic running cost
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SETSNOPTPARAM - Helper function to set SNOPT parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function setSNOPTParam(paramstring, value)
    snset([paramstring, '=', num2str(value)]);
end