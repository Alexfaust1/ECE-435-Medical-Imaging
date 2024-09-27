% Function that performs the soft thresholding operation for the gradient
% descent algorithm:
function sign_x = softThreshold(input, threshold)
    % Specify the input image data to perform the soft thresholding on.
    % Thershold value gives the value to distinguish its two outputs:
    %
    % NOTE: Function only works for real numbers because the equation in
    % notes gives something different  for complex valued inputs
    temp = input .* (abs(input) > threshold);
    sign_x = sign(temp).*(abs(input) - threshold);
end