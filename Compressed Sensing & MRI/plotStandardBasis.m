% Function to plot the standard basis plots for each wavelet domain. (one
% point in the domain)
function plotStandardBasis(coefficients, wavelet_name)

    % Compute the wavelet tranform and find approximate coefficients:
    [C, S] = wavedec2(coefficients, 2, wavelet_name);
    Approx2 = appcoef2(C, S, wavelet_name, 1);

    % Compute FT of the wavelet coefficients - basis - and the
    % reconstructed image:
    basis_fft = fftshift(fft2(coefficients));
    Approx2_fft = fftshift(fft2(Approx2));
    
    % Decode the wavelet_name variable into its string for wavelet
    % functions:
    titlename = WaveletName(wavelet_name);

    % Create the image plots:
    figure;
    subplot(2, 2, 1);
    imagesc(coefficients);
    title("Standard Basis");
    colormap("gray");
    axis("square");
    
    subplot(2, 2, 2);
    imagesc(abs(Approx2));
    title("Reconstructed Image");
    colormap("gray");
    axis("square");
    
    subplot(2, 2, 3);
    imagesc(20*log10(abs(basis_fft) + 1));
    title("Fourier Transform of Original Image");
    colormap("gray");
    axis("square");
    
    subplot(2, 2, 4);
    imagesc(20*log10(abs(Approx2_fft) + 1));
    title("Fourier Transform of Reconstructed Image");
    colormap("gray");
    axis("square");
    colorbar;

    % Create plot title:
    sgtitle("Standard Basis for " + titlename + " Wavelet Transform")
   
end