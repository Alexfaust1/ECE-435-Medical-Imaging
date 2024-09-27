% Create basic function to plot the original image and the compressed image
% in the wavelet domain with s% compression
function plotCompressedImage(Image, wavelet_name, n, s)

    % Pre allocate the reconstructed image
    reconstructedImage = zeros(256, 256, length(s));

    % For every s, reconstruct the image
    for i = 1:length(s)
        [C, S] = wavedec2(Image, n, wavelet_name);
        C_reduced = reduceS(C, s(i));
        reconstructedImage(:, :, i) = waverec2(C_reduced, S, wavelet_name);
    end

    % Plot the image and display the MSE:
    figure;
    subplot(1, length(s) + 1, 1);
    imagesc(Image);
    title("Original Image");
    colormap("gray");
    axis("square");
    disp("Using the " + n + "-level " + WaveletName(wavelet_name) + " Wavelet Transform");
    
    % Create the reconstructed image plots:
    for i = 1:length(s)
        subplot(1, length(s) + 1, i + 1);
        imagesc(reconstructedImage(:, :, i));
        title("Image reconstructed with " + s(i) + "% reduction");
        colormap("gray");
        axis("square");
        disp("Mean Squared Error with s = " + s(i) + "%: " + immse(Image,reconstructedImage(:, :, i)));
    end
    disp(newline);
    sgtitle("Image reconstructed with " + n + "-level " + WaveletName(wavelet_name) + " Wavelet Transform");

end