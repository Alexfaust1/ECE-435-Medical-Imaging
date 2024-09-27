% Function to plot the Wavelet plot in the wavelet domain. Produce the
% level 1 and 2 horizontal, vertical and diagonal detail coefficients
function waveletTransformDisplay(input_image, wavelet_name)

    % Take 2D wavelet transform
    [C, S] = wavedec2(input_image, 2, wavelet_name);

    % Obtain detail coefficients via detcoef2
    [H1, V1, D1] = detcoef2("all", C, S, 1);        % Level 1
    [H2, V2, D2] = detcoef2("all", C, S, 2);        % Level 2

    % Extract 2D approximation coefficients of level 2:
    A2 = appcoef2(C, S, wavelet_name, 2);

    % Diyplaying the approximation image as the stand-alone image for each
    % group of 6 reduced coefficient images
    figure; 
    imagesc(A2);
    colormap gray;
    title(wavelet_name);
    
    % Plot the rest of the subplots:
    figure;
    subplot(2,3,4);
    imagesc(H1);
    colormap gray;
    title('Horizontal Detail Coefficients | Level 1');
    
    subplot(2,3,5);
    imagesc(V1);
    title('Vertical Detail Coefficients | Level 1');
    
    subplot(2,3,6);
    imagesc(D1);
    title('Diagonal Detail Coefficients | Level 1');
    
    subplot(2,3,1);
    imagesc(H2);
    title('Horizontal Detail Coefficients | Level 2');

    subplot(2,3,2);
    imagesc(V2);
    title('Vertical Detail Coefficients | Level 2');
    
    subplot(2,3,3);
    imagesc(D2);
    title('Diagonal Detail Coefficients | Level 2');

    sgtitle(wavelet_name);
end