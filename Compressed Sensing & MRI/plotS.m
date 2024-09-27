% Function to create the subplots for plotting the comparison between the
% guiding image and the s% removed frequencies
function plotS(guiding_image, s)

    % Compute 2D FFT of the guiding image
    FT = fftshift(fft2(guiding_image));

    % Remove the s% lowest magnitudes using reduce s function:
    reduced_magnitude = reduceS(FT, s);

    % Perform the image reconstruction with an IFFT:
    compressed_image = abs(ifft2(reduced_magnitude));

    % Compute MSE with the MSE function:
    MSE = immse(abs(guiding_image), compressed_image);

    % Create the guiding image and reconstructed image subplots:
    figure;
    subplot(1,2,1);
    imagesc(guiding_image);
    title("Guiding Image");
    colormap("gray");
    axis("square");
    
    subplot(1,2,2);
    imagesc(compressed_image);
    title("Reconstructed Image");
    colormap("gray");
    axis("square");
    
    % Display the output MSE for the s:
    sgtitle("Compression with s = " + s + "% resulting in MSE = " + MSE);

end