classdef EKF < kalman.AbsKF
    %EKF Extended Kalman filter
    %   
    %   System model:
    %   - x[n] = f(x[n-1], u[n-1])
    %   - z[n] = h(x[n])
    %   
    %   Author: Dan Oates (WPI Class of 2020)
    
    properties (Access = protected)
        f;  % State func [Rn, Rm -> Rn]
        h;  % Output func [Rn -> Rp]
        fx; % State Jacobian func [Rn, Rm -> Rnn]
        fu; % Input Jacobian func [Rn, Rm -> Rnm]
        hx; % Output Jacobian func [Rn -> Rpn]
    end
    
    methods (Access = public)
        function obj = EKF(x_est, cov_x, cov_u, cov_z, f, h, fx, fu, hx)
            %obj = EKF(x_est, cov_x, cov_u, cov_z, f, h, fx, fu, hx)
            %   Construct Extended Kalman filter
            %   - x_est = State estimate [n x 1]
            %   - cov_x = State cov [n x n]
            %   - cov_u = Input cov [m x m]
            %   - cov_z = Output cov [p x p]
            %   - f = State function [Rn, Rm -> Rn]
            %   - h = Output function [Rn -> Rp]
            %   - fx = State Jacobian function [Rn, Rm -> Rnn]
            %   - fu = Input Jacobian function [Rn, Rm -> Rnm]
            %   - hx = Output Jacobian function [Rn -> Rpn]
            %   For n > 1 outputs, make cov_z, h, and hx [n x 1] cells.
            obj@kalman.AbsKF(x_est, cov_x, cov_u, cov_z);
            obj.f = f;
            obj.h = obj.to_cell(h);
            obj.fx = fx;
            obj.fu = fu;
            obj.hx = obj.to_cell(hx);
            obj.jac_zx = cell(obj.n_out, 1);
        end
        
        function x_est = predict(obj, u)
            %x_est = PREDICT(obj, u)
            %   Prediction step
            %   - u = Input vector [m x 1]
            %   - x_est = Predicted state [n x 1]
            obj.jac_xx = obj.fx(obj.x_est, u);
            obj.jac_xu = obj.fu(obj.x_est, u);
            x_est = predict@kalman.AbsKF(obj, u);
        end
        
        function x_est = correct(obj, z, i)
            %x_est = CORRECT(obj, z, i)
            %   Correction step
            %   - z = Output vector [p x 1]
            %   - i = Output index [int, def = 1]
            %   - x_est = Corrected state [n x 1]
            if nargin < 3, i = 1; end
            obj.jac_zx{i} = obj.hx{i}(obj.x_est);
            x_est = correct@kalman.AbsKF(obj, z, i);
        end
    end
    
    methods (Access = protected)
        function x = predict_x(obj, u)
            %x = PREDICT_X(obj, u)
            %   Predict state
            %   - u = Input vector [m x 1]
            %   - x = Predicted state [n x 1]
            x = obj.f(obj.x_est, u);
        end
        
        function z = predict_z(obj, i)
            %z = PREDICT_Z(obj, x)
            %   Predict output
            %   - i = Output index [1...n_out]
            %   - z = Predicted output [p x 1]
            z = obj.h{i}(obj.x_est);
        end
    end
end