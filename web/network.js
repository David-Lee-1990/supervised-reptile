(function() {

    var BATCHNORM_EPSILON = 0.001;

    function Variable(initialization) {
        this.initialization = initialization;
        this.value = initialization;
        this.gradient = zeros(initialization.length);
    }

    Variable.prototype.backward = function(outgrad) {
        for (var i = 0; i < outgrad.length; ++i) {
            this.gradient[i] += outgrad[i];
        }
    };

    Variable.prototype.clearGrad = function() {
        for (var i = 0; i < this.gradient.length; ++i) {
            this.gradient[i] = 0;
        }
    };

    function relu(input) {
        var inputs = input.value;
        var result = [];
        for (var i = 0; i < inputs.length; ++i) {
            if (inputs[i] < 0) {
                result.push(0);
            } else {
                result.push(inputs[i]);
            }
        }
        return {
            value: result,
            backward: function(outgrad) {
                var ingrad = [];
                for (var i = 0; i < inputs.length; ++i) {
                    if (inputs[i] < 0) {
                        ingrad.push(0);
                    } else {
                        ingrad.push(outgrad[i])
                    }
                }
                input.backward(ingrad);
            }
        };
    }

    function scale(input, scalar) {
        var inputs = input.value;
        var result = [];
        for (var i = 0; i < inputs.length; ++i) {
            result.push(inputs[i] * scalar);
        }
        return {
            value: result,
            backward: function(outgrad) {
                var ingrad = [];
                for (var i = 0; i < inputs.length; ++i) {
                    ingrad[i] = outgrad[i] * scalar;
                }
                input.backward(ingrad);
            }
        };
    }

    function pool(input, func) {
        var poolVar = new Variable(input.value);
        var result = func(poolVar);
        return {
            value: result.value,
            backward: function(outgrad) {
                poolVar.clearGrad();
                result.backward(outgrad);
                input.backward(poolVar.gradient);
            }
        };
    }

    function rsqrt(input, epsilon) {
        var result = [];
        var inputs = input.value;
        for (var i = 0; i < inputs.length; ++i) {
            result.push(1 / Math.sqrt(inputs[i] + epsilon));
        }
        return {
            value: result,
            backward: function(outgrad) {
                var ingrad = [];
                for (var i = 0; i < inputs.length; ++i) {
                    ingrad.push(outgrad[i] * -0.5 * Math.pow(inputs[i] + epsilon, -1.5));
                }
                input.backward(ingrad);
            }
        };
    }

    function square(input) {
        var result = [];
        var inputs = input.value;
        for (var i = 0; i < inputs.length; ++i) {
            result.push(inputs[i] * inputs[i]);
        }
        return {
            value: result,
            backward: function(outgrad) {
                var ingrad = [];
                for (var i = 0; i < inputs.length; ++i) {
                    ingrad.push(outgrad[i] * 2 * inputs[i]);
                }
                input.backward(ingrad);
            }
        };
    }

    function channelMeans(input, numChannels) {
        var result = [];
        var inputs = input.value;
        for (var i = 0; i < inputs.length; ++i) {
            result[i % numChannels] = (result[i % numChannels] || 0) + inputs[i];
        }
        var scale = numChannels / inputs.length;
        for (var i = 0; i < result.length; ++i) {
            result[i] *= scale;
        }
        return {
            value: result,
            backward: function(outgrad) {
                var ingrad = [];
                for (var i = 0; i < inputs.length; ++i) {
                    ingrad.push(outgrad[i % numChannels] * scale);
                }
                input.backward(ingrad);
            }
        };
    }

    function scaleChannels(input, scales) {
        var result = [];
        var inputs = input.value;
        var scaleVals = scales.value;
        for (var i = 0; i < inputs.length; ++i) {
            result.push(inputs[i] * scaleVals[i % scaleVals.length]);
        }
        return {
            value: result,
            backward: function(outgrad) {
                var ingrad = [];
                var scaleGrad = zeros(inputs.length);
                for (var i = 0; i < inputs.length; ++i) {
                    ingrad.push(outgrad[i] * scaleVals[i % scaleVals.length]);
                    scaleGrad[i % scaleVals.length] += inputs[i] * outgrad[i];
                }
                scales.backward(scaleGrad);
                input.backward(ingrad);
            }
        };
    }

    function addChannels(input, biases) {
        var result = [];
        var inputs = input.value;
        var biasVals = biases.value;
        for (var i = 0; i < inputs.length; ++i) {
            result.push(inputs[i] + biasVals[i % biasVals.length]);
        }
        return {
            value: result,
            backward: function(outgrad) {
                var biasGrad = zeros(biasVals.length);
                for (var i = 0; i < inputs.length; ++i) {
                    biasGrad[i % biasVals.length] += outgrad[i];
                }
                biases.backward(biasGrad);
                input.backward(outgrad);
            }
        };
    }

    function batchNorm(input, biases, numChannels) {
        return pool(input, function(input) {
            var centered = addChannels(input, scale(channelMeans(input, numChannels), -1));
            var divisor = rsqrt(channelMeans(square(centered), numChannels), BATCHNORM_EPSILON);
            var normalized = scaleChannels(centered, divisor);
            return addChannels(normalized, biases);
        });
    }

    function dense(input, weights, numInputs, numOutputs) {
        var inputs = input.value;
        var weightVals = weights.value;
        var batchSize = inputs.length / numInputs;
        var result = [];
        for (var i = 0; i < batchSize; ++i) {
            var offset = i * numInputs;
            for (var j = 0; j < numOutputs; ++j) {
                var sum = 0;
                for (var k = 0; k < numInputs; ++k) {
                    sum += inputs[offset + k] * weightVals[k*numOutputs + j];
                }
                result.push(sum);
            }
        }
        return {
            value: result,
            backward: function(outgrad) {
                var weightGrad = zeros(weightVals.length);
                var inputGrad = zeros(inputs.length);
                var outgradIdx = 0;
                for (var i = 0; i < batchSize; ++i) {
                    var offset = i * numInputs;
                    for (var j = 0; j < numOutputs; ++j) {
                        var grad = outgrad[outgradIdx++];
                        for (var k = 0; k < numInputs; ++k) {
                            inputGrad[offset + k] += grad * weightVals[k*numOutputs + j];
                            weightGrad[k*numOutputs + j] += grad * inputs[offset + k];
                        }
                    }
                }
                input.backward(inputGrad);
                weights.backward(weightGrad);
            }
        };
    }

    function logSoftmax(input, numChannels) {
        var inputs = input.value;
        var batchSize = inputs.length / numChannels;
        var result = [];
        var logsumexps = [];
        for (var i = 0; i < batchSize; ++i) {
            var offset = i * numChannels;
            var max = -Infinity;
            for (var j = 0; j < numChannels; ++j) {
                max = Math.max(max, inputs[offset + j]);
            }
            var expSum = 0;
            for (var j = 0; j < numChannels; ++j) {
                expSum += Math.exp(inputs[offset + j] - max);
            }
            var lse = Math.log(expSum) + max;
            logsumexps.push(lse);
            for (var j = 0; j < numChannels; ++j) {
                result.push(inputs[offset + j] - lse);
            }
        }
        return {
            value: result,
            backward: function(outgrad) {
                var ingrad = zeros(inputs.length);
                for (var i = 0; i < batchSize; ++i) {
                    var offset = i * numChannels;
                    var gradSum = 0;
                    for (var j = 0; j < numChannels; ++j) {
                        gradSum += outgrad[offset + j];
                    }
                    for (var j = 0; j < numChannels; ++j) {
                        var normalizeGrad = Math.exp(inputs[offset + j] - logsumexps[i]);
                        ingrad[offset + j] = 1 * outgrad[offset + j];
                        ingrad[offset + j] -= normalizeGrad * gradSum;
                    }
                }
                input.backward(ingrad);
            }
        };
    }

    function zeros(length) {
        var res = [];
        for (var i = 0; i < length; ++i) {
            res.push(0);
        }
        return res;
    }

    var exported = {
        Variable: Variable,
        relu: relu,
        scale: scale,
        pool: pool,
        rsqrt: rsqrt,
        square: square,
        channelMeans: channelMeans,
        scaleChannels: scaleChannels,
        addChannels: addChannels,
        batchNorm: batchNorm,
        dense: dense,
        logSoftmax: logSoftmax,
        zeros: zeros
    };
    if ('undefined' !== typeof window) {
        window.neuralnet = (window.neuralnet || {});
        var keys = Object.keys(exported);
        for (var i = 0; i < keys.length; ++i) {
            window.neuralnet[keys[i]] = exported[keys[i]];
        }
    } else {
        module.exports = exported;
    }

})();