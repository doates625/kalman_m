classdef LKF < kalman.AbsKF
    %LKF Linear Kalman filter
    %   
    %   System model:
    %   x[n] = Fx*x[n-1] + Fu*u[n-1]
    %   z[n] = Hx*x[n]
    %   
    %   Author: Dan Oates (WPI Class of 2020)
    
    methods (Access = public)
        function obj = LKF(x_est, cov_x, cov_u, cov_z, Fx, Fu, Hx)
            %obj = KF(x_est, cov_x, cov_u, cov_z, Fx, Fu, Hx)
            %   Constuct Linear Kalman filter
            %   - x_est = State estimate [n x 1]
            %   - cov_x = State cov [n x n]
            %   - cov_u = Input cov [m x m]
            %   - cov_z = Output cov [p x p]
            %   - Fx = State matrix [n x n]
            %   - Fu = Input matrix [n x m]
            %   - Hx = Output matrix [p x n]
            %   For n > 1 outputs, make cov_z and Hx [n x 1] cells.
            obj@kalman.AbsKF(x_est, cov_x, cov_u, cov_z);
            obj.jac_xx = Fx;
            obj.jac_xu = Fu;
            obj.jac_zx = obj.to_cell(Hx);
        end
    end
    
    methods (Access = protected)
        function x = predict_x(obj, u)
            %x = PREDICT_X(obj, u)
            %   Predict state
            %   - u = Input vector [m x 1]
            %   - x = Predicted state [n x 1]
            x = obj.jac_xx * obj.x_est + obj.jac_xu * u;
        end
        
        function z = predict_z(obj, i)
            %z = PREDICT_Z(obj, i)
            %   Predict output
            %   - i = Output index [def = 1]
            %   - z = Predicted output [p x 1]
            if nargin < 2, i = 1; end
            z = obj.jac_zx{i} * obj.x_est;
        end
    end
end