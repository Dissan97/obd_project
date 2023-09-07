function [z_prime] = relu_prime(z)    
    z_prime = (z > 0) * 1;
end

