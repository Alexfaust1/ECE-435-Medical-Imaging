Center wavelength: 1310 nm
dz = 2.7um
Sampling frequency:97656.25 Hz
# Backgrounds in B-Scan: 175
Width of B-Scan: 1 mm
# Backgrounds in MScan1: 320
# Backgrounds in MScan40: 320


RAW files are binary files of little-endian unsigned 16-bit integers.
Each integer is the number of electrons stored on the CCD sensor at readtime.
 
There are 2048 pixels on the line camera, each corresponding to a wavelength.
Each A-Scans is 2048 entries in the RAW file.

That is, the first 2048 unsigned 16-bit integers are background scan 1
The next 2048 are background scan 2.
And on and on until you are done with backgrounds, after which the next 2048
are data scan 1. Then data scan 2. Etc.

Suggestion: Use fopen and fread to load the RAW file, then reshape so that
your columns are the right size.


Matrix L2K is a 2048x2048 matrix that relates power in lambda to power in k
Once you have your processed data ready in lambda domain, L2K*data is that
data in the k domain.