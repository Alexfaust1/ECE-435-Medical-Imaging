% Function that reduces the output of S for the s% wavelet deconstruction
function reduced_output = reduceS(input,s)
    % Get indicies of sorted elements:
    [~,lowest_index] = sort(abs(input(:)));

    % Remove the s% smallest indicies
    removeInd = lowest_index(1 : round((s/100) * length(input(:))));

    % Re-index the input by the non removed indicies
    var = input;
    var(removeInd) = 0;
    reduced_output = var;
end