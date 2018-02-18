const neuralnet = require('../network');
const conv2d = require('../conv');
const {assertClose} = require('./test');

function testAll() {
    testSimpleConv();
    testUnitStridedConv();
    testSimpleStridedConv();
    testComplexConv();
}

function testSimpleConv() {
    const kernel = [0.501973, 0.534646, -0.399617, 0.325887, 0.315172, -0.476188, -0.359616, -0.384478, -0.196545];
    const inputs = [-0.101130, 1.076994, 0.466925, -0.380290];
    const outputs = [-0.649503, 0.284781, -0.156203, 0.557353];
    const upstream = [0.077384, 0.097113, -0.347073, -1.011891];
    const inputGrad = [-0.637467, -0.408549, -0.503825, -0.206194];
    const kernelGrad = [0.102333, -1.054700, -0.373796, -0.482298, 0.319519, 0.215331, 0.045344, -0.000798, -0.029428];
    testConv(kernel, inputs, outputs, upstream, inputGrad, kernelGrad, 2, 2, 1, 1);
}

function testUnitStridedConv() {
    const kernel = [-0.259365, -0.334450, 0.462496, 0.449582, -0.169502, 0.359014, -0.495809, 0.070711, 0.500750];
    const inputs = [-0.418408];
    const outputs = [0.070921];
    const upstream = [0.448628];
    const inputGrad = [-0.076044];
    const kernelGrad = [0.000000, 0.000000, 0.000000, 0.000000, -0.187710, 0.000000, 0.000000, 0.000000, 0.000000];
    testConv(kernel, inputs, outputs, upstream, inputGrad, kernelGrad, 1, 1, 1, 2);
}

function testSimpleStridedConv() {
    const kernel = [-0.123463, -0.130289, -0.480439,
                  -0.253978, 0.546918, 0.295094,
                  -0.213059, -0.190697, -0.449999];
    const inputs = [-1.279817, 0.527498,
                  0.758683, 0.277166];
    const outputs = [0.048181];
    const upstream = [0.159830];
    const inputGrad = [-0.019733, -0.020824, -0.040593, 0.087414];
    const kernelGrad = [-0.204553, 0.084310, 0.000000, 0.121260, 0.044299, 0.000000, 0.000000, 0.000000, 0.000000];
    testConv(kernel, inputs, outputs, upstream, inputGrad, kernelGrad, 2, 2, 1, 2);
}

function testComplexConv() {
    const kernel = [-0.141946, 0.233484, 0.131047, -0.075337, -0.056885, 0.259588, 0.075691, 0.143701, -0.100100, 0.010609, 0.136851, 0.035377, -0.174468, 0.146290, 0.038709, 0.224023, -0.191079, 0.139555, 0.218104, -0.097238, 0.213516, 0.212870, 0.255595, -0.257535, 0.233774, 0.069014, 0.147220, 0.205504, 0.185592, -0.097121, -0.005779, -0.136142, 0.139503, -0.076542, 0.268915, -0.220851, 0.219918, -0.135691, -0.022646, 0.052025, -0.111207, -0.209216, 0.263216, -0.104730, -0.000647, -0.027946, -0.245505, -0.086294, 0.111648, 0.048882, -0.152963, -0.161241, -0.251722, 0.051081, -0.097214, -0.259534, -0.117468, 0.134598, -0.060296, -0.187476, 0.069183, -0.177820, -0.260808, 0.046723, -0.003115, 0.161914, -0.123116, -0.195322, 0.193781, -0.267034, -0.153568, -0.144855, 0.006639, -0.179033, 0.045348, -0.156199, -0.229562, 0.179651, 0.165272, -0.089432, -0.175512, -0.046281, 0.156089, -0.101382, -0.116789, 0.151078, -0.017178, -0.209695, -0.263147, -0.110170, 0.165298, -0.127463, -0.124264, -0.153244, -0.143248, 0.115865, -0.026054, 0.114702, -0.239953, -0.016528, 0.003031, -0.104757, -0.256264, 0.237692, -0.067443, 0.097779, -0.048777, 0.132018, 0.148316, 0.077759, -0.041589, -0.043103, 0.211430, -0.076127, -0.183021, 0.107462, -0.116027, -0.165242, -0.138727, -0.127511, -0.121416, 0.113886, -0.248619, -0.157855, 0.209173, 0.120582, -0.268987, 0.052387, 0.194265, -0.107885, -0.024363, -0.269960, 0.091586, -0.100594, 0.215529, -0.188941, -0.109772, 0.127686, -0.121267, -0.095504, -0.231714, -0.154520, -0.265729, 0.250316, -0.155977, 0.188483, -0.088836, 0.124739, 0.262500, -0.205463, -0.240623, -0.104990, 0.248883, -0.014303, 0.077756, -0.217880, 0.069021, 0.254239, 0.243696, 0.035041, 0.221482, -0.141749, -0.163530, 0.137525, 0.128941, 0.125344, 0.225273, -0.122478, 0.220801, 0.043467, 0.027922, 0.237205, 0.074965, 0.169190, -0.124289, -0.060275, -0.035520, -0.192244, -0.089949, -0.145305];
    const inputs = [0.088152, -1.457493, -0.677335, 1.385346, -0.050314, 0.261057, 3.111908, 1.921729, -1.472107, 0.575417, 0.820531, 0.223083, -0.122377, 1.892949, 0.545836, 0.600628, -1.858793, 0.157140, -0.327055, -0.592249, 1.980689, -1.373783, -0.599625, 0.642983, -0.105883, -0.640657, -0.319734, 1.012364, 0.337197, -1.368432, -0.381286, -0.969247, -0.011637, 0.195885, -0.369695, 1.610209, 0.116471, 0.033103, -0.296365, 0.914737, 0.255127, -1.972945, 0.815941, 0.227066, 0.388921, 0.872884, 0.852717, -0.786585, -1.754977, 1.618268, 0.514467, -0.173250, 1.302264, 0.333753, -0.862668, 0.955281, 0.258117, 0.502796, 0.747338, 0.161898, -0.221227, -1.456534, 0.739725, -0.706098, -0.661105, 1.023111, 0.000994, 0.917040, -0.812022, -0.595177, -1.426389, -1.019603, 0.728447, -0.079995, 0.114371, 0.998254, -0.905327, -0.815747, 0.361915, -0.975415, 0.725262, -0.306614, 1.144791, 0.864791, -0.295998, 0.302538, -0.336840, -0.707301, 0.671399, 0.430836, -0.230974, -0.692599, 1.256307, -0.655178, -1.323539, 0.507360, 0.167643, 0.070550, -1.209311, -0.761843, -1.656044, 2.351540, 0.345370, 0.148679, -1.128499, -0.732706, -0.387073, -1.330970, -1.135800, -1.650412, 0.684539, -0.582704, -1.130293, -0.518286, 0.265839, 0.257644, 0.128373, -0.374568, 0.395829, -1.171561, -0.570868, -1.721977, -1.909619, -0.047027, -0.038852, -0.238520, -2.861591, -1.496120, 2.018493, -1.333468, 1.805086, 0.878986, 1.485746, -1.108068, 1.060928, 0.683460, -0.290281, -0.438184, -0.927186, 0.578405, -0.431776, 0.704038, 1.640742, -0.374823, -2.049794, -2.060761, -0.635816, -0.589886, -1.027318, 1.024152, 0.739327, -0.711166, -0.686138, -0.206224, 0.554367, -1.308055, 1.493898, -0.471107, -0.474859, -0.377400, -0.349482, 0.326429, 0.261963, -0.683193, -1.633183, 0.740709, 0.364062, -1.087360, -1.184710, -1.208543, -1.072565, -0.490396, -0.467164, -0.056005, -1.335155, -1.389091, -1.541859, -0.523355, 0.893209, 0.885714, -1.294316, -1.410916, 0.806707, -0.406302, 1.114517, -0.610827, -1.090186, 0.214819, -1.527347, 0.630027, 0.202690, -0.509037];
    const outputs = [-0.777604, -1.110567, 0.197954, 2.390106, -0.028291, 0.231061, 0.150760, -0.416802, 0.024617, 0.003097, 0.639323, -0.165765, -0.284590, -1.872204, 0.679274, 0.460756, -0.439972, -0.350246, 0.404049, -0.132759, -0.327878, -1.031172, -0.462482, 0.370434, 0.368367, 0.364359, -0.414007, -0.031376, 0.546345, -0.568325, -0.363367, 2.172801, 1.246897, 1.385930, -0.362991, -1.121943, -1.314405, -1.529844, -0.388257, 0.313027, 1.620800, -0.816746, 0.115431, -0.323588, 0.030228, -0.464531, 0.291271, -0.103287, 0.495692, 0.514834, 0.011660, 0.287934, -0.422748, 0.085381, 0.905030, 0.407436, -0.411467, -0.800923, 0.682328, -0.133151];
    const upstream = [1.530007, 0.159190, -0.308159, 0.398156, 2.041143, -0.147737, 0.543940, 0.099090, -0.302960, -0.584767, 0.991115, -0.620684, 0.290422, 0.790395, 0.616416, -0.519431, -0.089636, -0.532763, -0.247351, 1.857199, 0.010855, 0.958560, 0.166962, 0.900854, 0.014562, -1.015082, 0.179127, 0.208902, -1.123474, -0.949879, -0.438493, -0.414554, 1.182979, -0.588447, 0.534973, 1.001703, 0.833105, -0.387067, 0.604930, 0.258765, -0.904188, -0.569926, 0.076523, 2.334911, 0.462510, -1.933854, 1.036633, 0.054161, 0.302081, 0.503286, -1.828569, 0.610055, 0.575445, 1.321404, -0.623686, 0.545000, 0.004775, -0.816690, 0.219423, 0.265330];
    const inputGrad = [-0.366501, 0.346737, 0.406035, 0.157698, 0.656430, -0.058643, 0.444916, -0.163906, -0.110537, 0.130163, -0.445440, -0.996358, 0.050891, 0.090812, -0.193439, 0.115243, -0.381340, -0.020915, 0.153817, 0.609293, 0.094272, 0.123531, 0.224553, -0.380625, 0.170159, -0.179579, -0.238787, -0.447630, -0.602751, -0.036612, -0.082496, 0.008506, -0.121211, 0.306128, -0.567254, -0.200398, 0.115231, 0.091705, 0.024127, 0.062438, 0.022808, 0.050425, -0.040183, 0.136583, -0.252017, -0.176403, -0.002569, -0.035527, 0.169051, -0.191051, 0.345799, -0.997248, -0.284235, -0.463873, 0.150698, 0.037354, 0.801839, 0.442320, -0.185223, -0.519593, -0.040330, 0.318771, -0.210342, 0.166713, -0.402355, 0.121999, -0.146684, -0.689149, -0.167153, 0.241992, -0.223236, 0.240522, 0.101611, -0.512875, 0.217721, -0.200974, -0.179666, -0.104736, -0.236368, -0.090300, -0.210905, 0.012789, -0.708285, -0.204049, -0.113239, -0.288498, -0.281271, -0.220969, -0.136121, -0.080179, 0.211676, -0.210237, 0.427312, 0.200048, 0.091652, 0.186963, 0.134376, 0.089369, -0.346442, -0.034291, 0.397101, -0.009354, 0.412905, -0.113688, 0.449467, 0.182056, 0.086668, 0.374889, 0.196992, 0.199376, -0.149923, 0.014272, -0.648009, -0.614613, 0.017849, -0.113624, -0.788019, 0.257842, 0.039147, 0.035156, -0.294310, -0.507908, 0.264855, 0.231084, 0.277977, -0.211280, -0.153100, 0.228043, -0.387440, 0.271191, -0.140539, -0.521496, -0.366335, 0.030496, -0.022282, -0.099472, 0.267774, 0.353828, -0.427951, 0.588372, -0.093718, -0.808238, -0.510391, -0.649051, 0.388960, -0.492080, 0.231684, -0.307129, -0.517970, -0.125178, 0.483294, 0.910090, 0.178177, -1.065399, -0.556148, -0.461483, -0.746518, 0.559826, -0.674439, 0.294878, -0.088060, 0.931481, 0.738681, 0.355165, 0.703728, 0.293073, 0.235842, 0.762677, -0.319703, -0.527180, 0.115917, 0.078739, 0.210488, -0.456265, -0.576912, -0.325665, -0.411861, -0.209531, -0.136509, 0.079997, 0.321395, -0.686415, -0.564681, -0.468524, 0.398514, 0.233746, 0.067912, -0.686570, -0.276585, 0.166539, 0.119330, -0.087690];
    const kernelGrad = [4.941797, -2.257282, 0.455298, -4.601737, -4.154164, 0.211044, -2.101174, -0.182152, -4.245507, -2.387128, -1.481007, 0.919604, -1.143850, 1.464226, -2.318951, 2.588671, -1.308132, -0.858962, -2.637515, 0.846537, -0.113122, -2.272475, 0.306454, 3.865950, 1.373895, -3.887932, -0.967337, 1.734241, -5.688728, 1.406578, 5.803664, 2.375341, -1.150537, 3.892827, 6.736394, 5.903107, -1.563598, -0.241649, -2.763087, 5.675852, -0.790408, -3.300569, -1.114201, -1.371155, -2.471625, 0.211141, 0.970041, -0.748291, 1.465486, 1.197838, -0.199506, 1.003109, -1.063277, 1.144034, 3.231380, 4.975231, -0.428465, -2.198794, 0.695472, 0.238110, 7.293131, 0.488944, -3.361636, -1.389660, 1.417395, -0.586903, -0.702766, -1.696940, -4.621446, -5.113484, 3.892192, 2.539032, -1.301617, 2.559947, -1.854484, 1.553950, 1.966626, -0.662517, 4.384275, 3.516776, 4.409446, 0.116887, 0.539105, -3.320868, -1.688276, 1.248275, -2.637151, -0.102143, 0.716290, -2.232481, 3.508735, -1.099489, -3.633696, 8.569759, 0.226319, 2.417676, 0.044774, -1.287978, -2.900211, -6.679570, -0.657045, -1.281548, 2.812601, -0.012904, 0.717409, 3.029941, -1.328939, -1.808279, 0.694037, 0.281185, -2.737465, -1.198938, 1.461608, -3.679467, 2.746229, 0.061675, 0.805124, -0.062539, 0.118565, 5.681583, -2.850979, 0.748666, -1.811637, -1.324366, -5.576384, 3.817600, 0.392525, -2.484764, 3.150740, 2.634923, 1.274963, 1.063915, -1.030884, 1.300369, 0.538163, 0.186011, -0.964547, 0.091962, -1.432566, -0.853059, 4.643967, 3.192230, -2.577566, -2.361403, 1.368547, -1.454794, -1.608918, 1.030553, 0.945575, 1.934666, -3.971925, 0.240786, 1.011154, -1.567253, -2.631618, 1.472337, 1.000966, -1.438886, -2.560324, 0.753241, 0.443380, -0.325225, -0.821464, 0.495403, 0.455948, 1.035543, 0.993993, -0.423870, 0.209049, 0.402141, 1.162607, 0.107937, 0.324208, 0.129509, 1.889200, 0.001441, 0.497676, -1.241982, 0.143072, -1.082358];

    testConv(kernel, inputs, outputs, upstream, inputGrad, kernelGrad, 4, 6, 4, 2);
}

function testConv(kernel, inputs, outputs, upstream, inputGrad, kernelGrad,
                  inRows, inCols, inDepth, stride) {
    const kernelVar = new neuralnet.Variable(kernel);
    const inputVar = new neuralnet.Variable(inputs);
    const actualOutput = conv2d(inputVar, kernelVar, inRows, inCols, inDepth, stride);
    assertClose(outputs, actualOutput.value);
    actualOutput.backward(upstream);
    assertClose(inputVar.gradient, inputGrad);
    assertClose(kernelVar.gradient, kernelGrad);
}

testAll();
console.log('PASS');