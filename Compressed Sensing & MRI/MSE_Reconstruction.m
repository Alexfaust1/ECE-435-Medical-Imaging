% Function to compute the MSE reconstruction for the input image and the
% reconstructed image using the selected wavelet transform and removing the
% s% smallest coefficients:
function MSE_reconstructed = MSE_Reconstruction(Image, wavelet_name, n, s)

    % Preallocate 
    MSE_reconstructed = zeros(1,length(s));
    
    % For every iteration through the length of s perform a wavelet
    % transform and find the mean square error while reconstructing the
    % image along the way.
    for c = 1:length(s)

        % Wavelet transform
        [C,S] = wavedec2(Image,n,wavelet_name);

        % Remove the s smallest coefficients
        C_reduced = reduceS(C,s(c));

        % Reconstruct the image 
        reconstructedImage = waverec2(C_reduced,S,wavelet_name);

        % Compute Mean square error
        MSE_reconstructed(c) = immse(Image,reconstructedImage);
    end
    
end