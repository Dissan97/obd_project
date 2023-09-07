function [dz] = sigmoid_prime(z)
%SIGMOID_PRIME Summary of this function goes here
%   Detailed explanation goes here
    sig = sigmoid(z);
    dz = sig .* (1 - sig);
end

