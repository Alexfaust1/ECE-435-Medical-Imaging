% Function to compute the normalized mean squared error between the guiding
% image and the reconstructed image for the selected wavelet transform.
% Also removes the s% coefficients.
function NMSE = NMSE_Reconstruction(input,wavelet_domain,n,s)

    % Initialize NMSE vector
    NMSE = zeros(1,length(s));

    for c = 1:length(s)
        % take the wavelet transform
        [C,S] = wavedec2(input,n,wavelet_domain);

        % remove s% lowest coefficient values
        C_reduced = reduceS(C,s(c));

        % reconstruct the image
        recontrcuted = waverec2(C_reduced,S,wavelet_domain);

        % calcuate the normalized mean square error
        NMSE(c) = immse(input,recontrcuted)/mean(input(:).^2);
    end
    
end