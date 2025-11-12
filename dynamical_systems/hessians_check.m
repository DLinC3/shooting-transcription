%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% hessians_check - Verify analytical Hessians for cart-pole dynamics
%
% This function validates the analytical second-order derivatives (Hessians)
% of the cart-pole dynamics against numerical differentiation.
%
% The verification includes:
%   - fxx: State-state Hessians (d²f/dx_i dx_j)
%   - fux: Control-state Hessians (d²f/du_i dx_j)
%   - fuu: Control-control Hessians (d²f/du_i du_j)
%
% Two verification approaches:
%   1. Specific test points at critical configurations
%   2. Statistical verification over random samples
%
% Author: [Your name]
% Date: [Current date]
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function hessians_check()
    
    %% --- Initialize System Parameters ---
    param.mc = 10;   % Cart mass [kg]
    param.mp = 2;    % Pole mass [kg]
    param.l = 0.5;   % Pole half-length [m]
    param.g = 9.8;   % Gravitational acceleration [m/s²]
    param.b = 0.1;   % Cart friction coefficient
    param.d = 0.1;   % Pole friction coefficient
    
    %% --- Define Test Points at Critical Configurations ---
    % Test points: [x, theta, x_dot, theta_dot, u]
    test_points = [
        % Configuration 1: Near downward equilibrium
        [0, 0.1, 0, 0, 0];           
        
        % Configuration 2: Near upward equilibrium (inverted)
        [0, pi-0.1, 0, 0, 0];         
        
        % Configuration 3: General position with motion
        [0.5, pi/4, 0.1, 0.2, 5];     
        
        % Configuration 4: Horizontal pole with velocity
        [0, pi/2, 0.5, -0.3, -10];    
        
        % Configuration 5: Another general case
        [-0.3, 3*pi/4, -0.2, 0.4, 2]; 
    ];
    
    %% --- Run Verification at Test Points ---
    fprintf('===============================================\n');
    fprintf('Hessian Verification for Cart-Pole Dynamics\n');
    fprintf('===============================================\n\n');
    
    for test_idx = 1:size(test_points, 1)
        % Extract state and control
        x = test_points(test_idx, 1:4)';
        u = test_points(test_idx, 5);
        
        fprintf('Test Point %d:\n', test_idx);
        fprintf('State: [%.2f, %.2f, %.2f, %.2f], Control: %.2f\n', ...
                x(1), x(2), x(3), x(4), u);
        fprintf('-----------------------------------------------\n');
        
        % Compute analytical Hessians
        [fx_ana, fu_ana, fxx_ana, fux_ana, fuu_ana] = ...
            cartpole_grads_and_hessians_full(x, u, param);
        
        % Compute numerical Hessians via finite differences
        [fxx_num, fux_num, fuu_num] = compute_numerical_hessians(x, u, param);
        
        % Verify each Hessian component
        fprintf('Checking fxx (state-state Hessians):\n');
        check_fxx(fxx_ana, fxx_num);
        
        fprintf('\nChecking fux (control-state Hessians):\n');
        check_fux(fux_ana, fux_num);
        
        fprintf('\nChecking fuu (control-control Hessians):\n');
        check_fuu(fuu_ana, fuu_num);
        
        fprintf('\n');
    end
    
    %% --- Run Statistical Verification ---
    fprintf('===============================================\n');
    fprintf('Statistical Verification (100 random points)\n');
    fprintf('===============================================\n\n');
    run_statistical_verification(param);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% COMPUTE_NUMERICAL_HESSIANS - Calculate Hessians using finite differences
%
% Uses central difference formulas for numerical second derivatives.
%
% Inputs:
%   x     - State vector [4×1]
%   u     - Control input [scalar]
%   param - System parameters
% Outputs:
%   fxx_num - State-state Hessians [4×4×4]
%   fux_num - Control-state Hessians [1×4×4]
%   fuu_num - Control-control Hessians [1×1×4]
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [fxx_num, fux_num, fuu_num] = compute_numerical_hessians(x, u, param)
    nX = 4;  % Number of states
    nU = 1;  % Number of controls
    
    % Step sizes for numerical differentiation
    % Small enough for accuracy, large enough to avoid round-off errors
    h_x = 1e-5;  % Step size for state perturbations
    h_u = 1e-5;  % Step size for control perturbations
    
    % Initialize Hessian tensors
    fxx_num = zeros(nX, nX, nX);  % fxx(i,j,k) = d²f_k/dx_i dx_j
    fux_num = zeros(nU, nX, nX);  % fux(i,j,k) = d²f_k/du_i dx_j
    fuu_num = zeros(nU, nU, nX);  % fuu(i,j,k) = d²f_k/du_i du_j
    
    %% --- Compute State-State Hessians (fxx) ---
    % For each output dimension k
    for k = 1:nX
        % For each pair of input states (i,j)
        for i = 1:nX
            for j = 1:nX
                fxx_num(i,j,k) = compute_fxx_element(x, u, param, i, j, k, h_x);
            end
        end
        % Enforce symmetry (since mixed partials are equal)
        fxx_num(:,:,k) = 0.5 * (fxx_num(:,:,k) + fxx_num(:,:,k)');
    end
    
    %% --- Compute Control-State Hessians (fux) ---
    for k = 1:nX
        for i = 1:nU
            for j = 1:nX
                fux_num(i,j,k) = compute_fux_element(x, u, param, i, j, k, h_u, h_x);
            end
        end
    end
    
    %% --- Compute Control-Control Hessians (fuu) ---
    % For cart-pole, dynamics are linear in u, so fuu should be zero
    for k = 1:nX
        for i = 1:nU
            for j = 1:nU
                fuu_num(i,j,k) = compute_fuu_element(x, u, param, i, j, k, h_u);
            end
        end
    end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% COMPUTE_FXX_ELEMENT - Compute single element of state-state Hessian
%
% Uses 4-point central difference formula:
% d²f/dx_i dx_j ≈ [f(+,+) - f(+,-) - f(-,+) + f(-,-)] / (4h²)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function fxx_ij_k = compute_fxx_element(x, u, param, i, j, k, h)
    % Create perturbed states
    x_pp = x; x_pp(i) = x_pp(i) + h; x_pp(j) = x_pp(j) + h;  % (+,+)
    x_pm = x; x_pm(i) = x_pm(i) + h; x_pm(j) = x_pm(j) - h;  % (+,-)
    x_mp = x; x_mp(i) = x_mp(i) - h; x_mp(j) = x_mp(j) + h;  % (-,+)
    x_mm = x; x_mm(i) = x_mm(i) - h; x_mm(j) = x_mm(j) - h;  % (-,-)
    
    % Evaluate dynamics at perturbed points
    f_pp = cartpole_dynamics(0, x_pp, u, param);
    f_pm = cartpole_dynamics(0, x_pm, u, param);
    f_mp = cartpole_dynamics(0, x_mp, u, param);
    f_mm = cartpole_dynamics(0, x_mm, u, param);
    
    % Central difference approximation
    fxx_ij_k = (f_pp(k) - f_pm(k) - f_mp(k) + f_mm(k)) / (4*h*h);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% COMPUTE_FUX_ELEMENT - Compute single element of control-state Hessian
%
% Mixed partial derivative using central differences
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function fux_ij_k = compute_fux_element(x, u, param, i, j, k, h_u, h_x)
    % Create perturbed controls and states
    u_p = u; u_p(i) = u_p(i) + h_u;  % u + h_u
    u_m = u; u_m(i) = u_m(i) - h_u;  % u - h_u
    x_p = x; x_p(j) = x_p(j) + h_x;  % x + h_x
    x_m = x; x_m(j) = x_m(j) - h_x;  % x - h_x
    
    % Evaluate at four corner points
    f_up_xp = cartpole_dynamics(0, x_p, u_p, param);  % (+u, +x)
    f_up_xm = cartpole_dynamics(0, x_m, u_p, param);  % (+u, -x)
    f_um_xp = cartpole_dynamics(0, x_p, u_m, param);  % (-u, +x)
    f_um_xm = cartpole_dynamics(0, x_m, u_m, param);  % (-u, -x)
    
    % Central difference for mixed partial
    fux_ij_k = (f_up_xp(k) - f_up_xm(k) - f_um_xp(k) + f_um_xm(k)) / (4*h_u*h_x);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% COMPUTE_FUU_ELEMENT - Compute single element of control-control Hessian
%
% For scalar control (cart-pole case), this is just the second derivative
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function fuu_ij_k = compute_fuu_element(x, u, param, i, j, k, h)
    if i == j
        % Diagonal element: d²f/du²
        u_p = u; u_p(i) = u_p(i) + h;   % u + h
        u_m = u; u_m(i) = u_m(i) - h;   % u - h
        u_0 = u;                         % u
        
        % Three-point central difference for second derivative
        f_p = cartpole_dynamics(0, x, u_p, param);
        f_m = cartpole_dynamics(0, x, u_m, param);
        f_0 = cartpole_dynamics(0, x, u_0, param);
        
        fuu_ij_k = (f_p(k) - 2*f_0(k) + f_m(k)) / (h*h);
    else
        % Off-diagonal element (for multi-input systems)
        % Not used in cart-pole (single input), but included for completeness
        u_pp = u; u_pp(i) = u_pp(i) + h; u_pp(j) = u_pp(j) + h;
        u_pm = u; u_pm(i) = u_pm(i) + h; u_pm(j) = u_pm(j) - h;
        u_mp = u; u_mp(i) = u_mp(i) - h; u_mp(j) = u_mp(j) + h;
        u_mm = u; u_mm(i) = u_mm(i) - h; u_mm(j) = u_mm(j) - h;
        
        f_pp = cartpole_dynamics(0, x, u_pp, param);
        f_pm = cartpole_dynamics(0, x, u_pm, param);
        f_mp = cartpole_dynamics(0, x, u_mp, param);
        f_mm = cartpole_dynamics(0, x, u_mm, param);
        
        fuu_ij_k = (f_pp(k) - f_pm(k) - f_mp(k) + f_mm(k)) / (4*h*h);
    end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CHECK_FXX - Verify state-state Hessian accuracy
%
% Compares analytical and numerical Hessians element by element
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function check_fxx(fxx_ana, fxx_num)
    nX = 4;
    max_error = 0;
    max_error_info = struct();
    
    % Check each element of the Hessian tensor
    for k = 1:nX  % Output dimension
        for i = 1:nX  % First input dimension
            for j = i:nX  % Second input dimension (upper triangle only due to symmetry)
                ana_val = fxx_ana(i,j,k);
                num_val = fxx_num(i,j,k);
                
                % Only check non-negligible elements
                if abs(ana_val) > 1e-10 || abs(num_val) > 1e-10
                    abs_error = abs(ana_val - num_val);
                    rel_error = abs_error / (max(abs(ana_val), abs(num_val)) + 1e-10);
                    
                    % Track maximum error
                    if abs_error > max_error
                        max_error = abs_error;
                        max_error_info.k = k;
                        max_error_info.i = i;
                        max_error_info.j = j;
                        max_error_info.ana = ana_val;
                        max_error_info.num = num_val;
                        max_error_info.rel = rel_error;
                    end
                    
                    % Warn about large errors
                    if abs_error > 1e-3
                        fprintf('  WARNING: fxx(%d,%d,%d): ana=%.6f, num=%.6f, err=%.2e (%.1f%%)\n', ...
                                i, j, k, ana_val, num_val, abs_error, rel_error*100);
                    end
                end
            end
        end
    end
    
    % Report results
    if max_error > 0
        fprintf('  Max error: fxx(%d,%d,%d) = %.2e (%.1f%% relative)\n', ...
                max_error_info.i, max_error_info.j, max_error_info.k, ...
                max_error, max_error_info.rel*100);
    end
    
    % Overall assessment
    tolerance = 1e-3;
    if max_error < tolerance
        fprintf('  ✓ All fxx elements match within tolerance (max error: %.2e)\n', max_error);
    else
        fprintf('  ✗ Some fxx elements have large errors!\n');
    end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CHECK_FUX - Verify control-state Hessian accuracy
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function check_fux(fux_ana, fux_num)
    nX = 4;
    nU = 1;
    max_error = 0;
    max_error_info = struct();
    
    % Check each element
    for k = 1:nX      % Output dimension
        for i = 1:nU  % Control dimension
            for j = 1:nX  % State dimension
                ana_val = fux_ana(i,j,k);
                num_val = fux_num(i,j,k);
                
                % Only check non-negligible elements
                if abs(ana_val) > 1e-10 || abs(num_val) > 1e-10
                    abs_error = abs(ana_val - num_val);
                    rel_error = abs_error / (max(abs(ana_val), abs(num_val)) + 1e-10);
                    
                    % Track maximum error
                    if abs_error > max_error
                        max_error = abs_error;
                        max_error_info.k = k;
                        max_error_info.i = i;
                        max_error_info.j = j;
                        max_error_info.ana = ana_val;
                        max_error_info.num = num_val;
                        max_error_info.rel = rel_error;
                    end
                    
                    % Warn about large errors
                    if abs_error > 1e-3
                        fprintf('  WARNING: fux(%d,%d,%d): ana=%.6f, num=%.6f, err=%.2e (%.1f%%)\n', ...
                                i, j, k, ana_val, num_val, abs_error, rel_error*100);
                    end
                end
            end
        end
    end
    
    % Report results
    if max_error > 0
        fprintf('  Max error: fux(%d,%d,%d) = %.2e (%.1f%% relative)\n', ...
                max_error_info.i, max_error_info.j, max_error_info.k, ...
                max_error, max_error_info.rel*100);
    end
    
    % Overall assessment
    tolerance = 1e-3;
    if max_error < tolerance
        fprintf('  ✓ All fux elements match within tolerance (max error: %.2e)\n', max_error);
    else
        fprintf('  ✗ Some fux elements have large errors!\n');
    end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CHECK_FUU - Verify control-control Hessian accuracy
%
% For cart-pole, fuu should be zero since dynamics are linear in control
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function check_fuu(fuu_ana, fuu_num)
    nX = 4;
    nU = 1;
    max_error = 0;
    
    % Check all elements (should be near zero)
    for k = 1:nX
        for i = 1:nU
            for j = 1:nU
                ana_val = fuu_ana(i,j,k);
                num_val = fuu_num(i,j,k);
                abs_error = abs(ana_val - num_val);
                
                if abs_error > max_error
                    max_error = abs_error;
                end
                
                % Warn if not near zero (dynamics should be linear in u)
                if abs_error > 3e-5
                    fprintf('  WARNING: fuu(%d,%d,%d): ana=%.6f, num=%.6f, err=%.2e\n', ...
                            i, j, k, ana_val, num_val, abs_error);
                end
            end
        end
    end
    
    % Assessment
    tolerance = 3e-5;  % Slightly larger tolerance due to numerical errors
    if max_error < tolerance
        fprintf('  ✓ All fuu elements are near zero as expected (max: %.2e)\n', max_error);
    else
        fprintf('  ✗ fuu should be zero for cart-pole (linear in control)!\n');
    end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% RUN_STATISTICAL_VERIFICATION - Statistical test over random samples
%
% Tests Hessian accuracy over many random state/control combinations
% to ensure robustness of the analytical derivatives
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function run_statistical_verification(param)
    n_tests = 100;  % Number of random tests
    nX = 4;
    
    % Initialize error tracking
    errors_fxx = zeros(n_tests, 1);
    errors_fux = zeros(n_tests, 1);
    errors_fuu = zeros(n_tests, 1);
    
    % Run tests
    for test = 1:n_tests
        % Generate random state and control with reasonable scales
        x = randn(4, 1) .* [1;      % Position: ±1 m
                            pi/2;    % Angle: ±π/2 rad
                            1;       % Velocity: ±1 m/s
                            2];      % Angular velocity: ±2 rad/s
        u = 20 * randn(1, 1);       % Control: ±20 N
        
        % Compute analytical Hessians
        [~, ~, fxx_ana, fux_ana, fuu_ana] = ...
            cartpole_grads_and_hessians_full(x, u, param);
        
        % Compute numerical Hessians
        [fxx_num, fux_num, fuu_num] = compute_numerical_hessians(x, u, param);
        
        % Record maximum errors for this test
        errors_fxx(test) = max(abs(fxx_ana(:) - fxx_num(:)));
        errors_fux(test) = max(abs(fux_ana(:) - fux_num(:)));
        errors_fuu(test) = max(abs(fuu_ana(:) - fuu_num(:)));
    end
    
    %% --- Report Statistics ---
    fprintf('fxx errors: mean=%.2e, std=%.2e, max=%.2e\n', ...
            mean(errors_fxx), std(errors_fxx), max(errors_fxx));
    fprintf('fux errors: mean=%.2e, std=%.2e, max=%.2e\n', ...
            mean(errors_fux), std(errors_fux), max(errors_fux));
    fprintf('fuu errors: mean=%.2e, std=%.2e, max=%.2e\n', ...
            mean(errors_fuu), std(errors_fuu), max(errors_fuu));
    
    %% --- Overall Assessment ---
    threshold_fxx_fux = 1e-3;
    threshold_fuu = 3e-4;  % Looser for fuu due to numerical issues
    
    if max(errors_fxx) < threshold_fxx_fux && ...
       max(errors_fux) < threshold_fxx_fux && ...
       max(errors_fuu) < threshold_fuu
        fprintf('\n✓✓✓ All Hessians PASSED verification! ✓✓✓\n');
    else
        fprintf('\n✗✗✗ Some Hessians FAILED verification! ✗✗✗\n');
        if max(errors_fxx) >= threshold_fxx_fux
            fprintf('  - fxx has errors above threshold (%.2e > %.2e)\n', ...
                    max(errors_fxx), threshold_fxx_fux);
        end
        if max(errors_fux) >= threshold_fxx_fux
            fprintf('  - fux has errors above threshold (%.2e > %.2e)\n', ...
                    max(errors_fux), threshold_fxx_fux);
        end
        if max(errors_fuu) >= threshold_fuu
            fprintf('  - fuu should be zero but exceeds threshold (%.2e > %.2e)\n', ...
                    max(errors_fuu), threshold_fuu);
        end
    end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CARTPOLE_DYNAMICS - Cart-pole system dynamics
%
% State: x = [x_cart, theta, x_dot, theta_dot]'
% Control: u = force on cart
%
% Returns: xdot = [x_dot, theta_dot, x_ddot, theta_ddot]'
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function xdot = cartpole_dynamics(t, x, u, param)
    % Extract parameters
    mc = param.mc;  % Cart mass
    mp = param.mp;  % Pole mass
    l = param.l;    % Pole half-length
    g = param.g;    % Gravity
    b = param.b;    % Cart friction
    d = param.d;    % Pole friction
    
    % Extract states
    x1 = x(1);  % Cart position
    x2 = x(2);  % Pole angle (0 = down)
    x3 = x(3);  % Cart velocity
    x4 = x(4);  % Pole angular velocity
    
    % Precompute trigonometric functions
    s = sin(x2); 
    c = cos(x2);
    
    % State derivatives
    xdot(1,:) = x3;  % dx1/dt = x3
    xdot(2,:) = x4;  % dx2/dt = x4
    
    % Cart acceleration (from equations of motion)
    xdot(3,:) = (u - b*x3 + d*x4*c/l + mp*s*(l*x4^2 + g*c)) / (mc + mp*s^2);
    
    % Pole angular acceleration
    xdot(4,:) = (-u*c + b*x3*c - d*(mc+mp)*x4/(mp*l) - mp*l*x4^2*c*s - (mc+mp)*g*s/(mp*l)) / ...
                (l*(mc + mp*s^2));
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CARTPOLE_GRADS_AND_HESSIANS_FULL - Analytical derivatives up to second order
%
% Computes Jacobians and Hessians of cart-pole dynamics analytically
%
% Outputs:
%   fx  - State Jacobian [4×4]
%   fu  - Control Jacobian [4×1]
%   fxx - State-state Hessian [4×4×4]
%   fux - Control-state Hessian [1×4×4]
%   fuu - Control-control Hessian [1×1×4] (zero for linear control)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [fx, fu, fxx, fux, fuu] = cartpole_grads_and_hessians_full(x, u, param)
    % First-order Jacobians
    [fx, fu] = cartpole_grads(0, x, u, param);

    % Parameters and variables
    mc = param.mc; mp = param.mp; l = param.l; g = param.g; b = param.b; d = param.d;

    x1 = x(1); x2 = x(2); x3 = x(3); x4 = x(4); %#ok<NASGU>
    s = sin(x2); c = cos(x2);

    nX = 4; nU = 1;
    fxx = zeros(nX, nX, nX);
    fux = zeros(nU, nX, nX);
    fuu = zeros(nU, nU, nX);  % Linear affine in u -> second-order is always 0

    % Common denominator: D = mc + mp*s^2
    D   = mc + mp*s^2;
    Dp  = 2*mp*s*c;                % dD/dtheta
    Dpp = 2*mp*(c^2 - s^2);        % d2D/dtheta2

    %-----------------------------
    % Third equation f3 = N3 / D
    %-----------------------------
    % Numerator N3(θ, x3, x4, u)
    %   N3 = u - b*x3 + d*x4*c/l + mp*s*(l*x4^2 + g*c)
    %   Written as: N3 = u - b*x3 + (d/l)*x4*c + mp*l*x4^2*s + mp*g*s*c
    N3   = u - b*x3 + (d/l)*x4*c + mp*l*x4^2*s + mp*g*s*c;
    N3p  =        -(d/l)*x4*s + mp*l*x4^2*c + mp*g*(c^2 - s^2);       % dN3/dθ
    N3pp =        -(d/l)*x4*c - mp*l*x4^2*s - 4*mp*g*s*c;              % d2N3/dθ2  (note -4*mp*g*s*c)

    % First-order and mixed derivatives w.r.t. x3, x4
    N3_x3    = -b;
    N3_x4    = (d/l)*c + 2*mp*l*x4*s;
    N3_px3   = 0;                                  % d^2N3/dθdx3
    N3_px4   = -(d/l)*s + 2*mp*l*x4*c;             % d^2N3/dθdx4
    N3_x4x4  = 2*mp*l*s;                           % d^2N3/dx4^2

    % Quotient rule for second derivatives:
    % If f = N(θ,·)/D(θ), then
    %   f_{θθ} = (N''/D) - (N*D''/D^2) - 2*(N'*D'/D^2) + 2*N*(D')^2/D^3
    %   f_{θx} = (N_{θx}*D - N_x*D') / D^2
    %   f_{xx} = N_{xx} / D  (if D is independent of x)

    % fxx(2,2,3) : d^2 f3 / dθ^2
    fxx(2,2,3) =  N3pp/D  - N3*Dpp/(D^2) - 2*N3p*Dp/(D^2) + 2*N3*(Dp^2)/(D^3);

    % fxx(2,3,3) = fxx(3,2,3) : d^2 f3 / dθ dx3
    val = (N3_px3*D - N3_x3*Dp) / (D^2);   % N3_px3=0
    fxx(2,3,3) = val; fxx(3,2,3) = val;

    % fxx(2,4,3) = fxx(4,2,3) : d^2 f3 / dθ dx4
    val = (N3_px4*D - N3_x4*Dp) / (D^2);
    fxx(2,4,3) = val; fxx(4,2,3) = val;

    % fxx(3,3,3) = 0   (f3 is linear in x3, D is independent of x3)
    % fxx(3,4,3) = 0   (same reason)
    % fxx(4,4,3) : d^2 f3 / dx4^2
    fxx(4,4,3) = N3_x4x4 / D;

    %-----------------------------
    % Fourth equation f4 = N4 / (l*D)
    %-----------------------------
    % Numerator N4(θ, x3, x4, u)
    %   N4 = -u*c + b*x3*c - d*(mc+mp)*x4/(mp*l) - mp*l*x4^2*c*s - (mc+mp)*g*s/(mp*l)
    N4   = -u*c + b*x3*c - d*(mc+mp)*x4/(mp*l) - mp*l*x4^2*c*s - (mc+mp)*g*s/(mp*l);
    N4p  =  u*s - b*x3*s - mp*l*x4^2*(c^2 - s^2) - (mc+mp)*g*c/(mp*l);          % dN4/dθ
    N4pp =  u*c - b*x3*c + 4*mp*l*x4^2*s*c + (mc+mp)*g*s/(mp*l);                 % d2N4/dθ2  (note +4*...*s*c)

    % First-order and mixed derivatives
    N4_x3    =  b*c;
    N4_x4    = - d*(mc+mp)/(mp*l) - 2*mp*l*x4*c*s;
    N4_px3   = - b*s;                           % d^2N4/dθdx3
    N4_px4   = - 2*mp*l*x4*(c^2 - s^2);         % d^2N4/dθdx4
    N4_x4x4  = - 2*mp*l*c*s;                    % d^2N4/dx4^2

    % Note: f4 = N4 / (l*D) = (1/l) * (N4 / D)
    % So we just apply the above formula with D, D', D'', then divide by l at the end.

    % fxx(2,2,4) : d^2 f4 / dθ^2
    fxx(2,2,4) = ( N4pp/D  - N4*Dpp/(D^2) - 2*N4p*Dp/(D^2) + 2*N4*(Dp^2)/(D^3) ) / l;

    % fxx(2,3,4) = fxx(3,2,4) : d^2 f4 / dθ dx3
    val = (N4_px3*D - N4_x3*Dp) / (D^2) / l;
    fxx(2,3,4) = val; fxx(3,2,4) = val;

    % fxx(2,4,4) = fxx(4,2,4) : d^2 f4 / dθ dx4
    val = (N4_px4*D - N4_x4*Dp) / (D^2) / l;
    fxx(2,4,4) = val; fxx(4,2,4) = val;

    % fxx(3,3,4) = 0  (linear in x3, D is independent of x3)
    % fxx(3,4,4) = 0
    % fxx(4,4,4) : d^2 f4 / dx4^2
    fxx(4,4,4) = (N4_x4x4 / D) / l;    % = -2*mp*c*s / D

    %-----------------------------
    % fux (second-order mixed derivatives w.r.t. u and x): by definition
    %-----------------------------
    % f3 = N3/D,  N3_u = 1,  N3_{ux} = 0 except x=theta? Only θ affects through D
    %   f_{u xj} = ∂/∂xj ( N_u/D ) = (N_{uxj}*D - N_u*D_xj)/D^2 = -(N_u*D_xj)/D^2
    %   D only depends on θ, so:
    %     fux(1,2,3) = -(1 * Dp)/D^2, other xj≠θ are 0
    fux(1,1,3) = 0;
    fux(1,2,3) = - Dp / (D^2);
    fux(1,3,3) = 0;
    fux(1,4,3) = 0;

    % f4 = N4/(lD),  N4_u = -c,  N4_{uxj}=0 (numerator w.r.t u has no x dependence)
    %   f_{u xj} = ∂/∂xj ( -c / (lD) ) = -( -s*δ_{j2}*D - c*D_{xj} ) / (lD^2)
    %   Only j=2(θ) is non-zero:  (s*D + c*Dp)/(l D^2) * ? Wait—let's compute more intuitively:
    %   For θ:  ∂θ[ -c/(lD) ] = ( s/(lD) ) + ( c*Dp/(lD^2) )
    %   But this is f_{θu}, and we want f_{uθ} (which is equivalent), so:
    fux(1,1,4) = 0;
    fux(1,2,4) = ( s*D + c*Dp ) / (l * D^2);
    fux(1,3,4) = 0;
    fux(1,4,4) = 0;

    % Symmetrize fxx(i,j,k) for each k
    for k = 1:nX
        fxx(:,:,k) = 0.5 * (fxx(:,:,k) + fxx(:,:,k)');
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% cartpole_grads - Analytical derivatives up to first order
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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