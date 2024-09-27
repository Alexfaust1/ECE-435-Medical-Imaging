% Function that takes the wavelet name and converts it to a string
function waveletName = WaveletName(wavelet_name)
    
    switch wavelet_name
        case "haar"
            waveletName = "Haar";

        case "db4"
            waveletName = "Daubuchies 4";

        case "coif3"
            waveletName = "Coiflet 3";

     end
end