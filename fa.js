// //For question 1
// let NI = 2;
// let NH = 4;
// let NO = 1;
// let tan = false;

// let learningValue = 0.7;
// let updateFreq = 0.3;
// let maxEpochs = 500000;

// let questionNumber = 1;

// //For question 2
// let NI = 4;
// let NH = 6;
// let NO = 1;
// let tan = true;

// let learningValue = 0.2;
// let updateFreq = 0.8;
// let maxEpochs = 10000;

// let questionNumber = 2;

//For question 3
let NI = 16;
let NH = 26;
let NO = 26;
let tan = false;

let learningValue = 0.15;
let updateFreq = 0.33;
let maxEpochs = 1500;

let questionNumber = 3;

let W1 = [...Array(NH)].map(e  => Array(NI).fill(0.0));
let dW1 = [...Array(NH)].map(e => Array(NI).fill(0.0));

let W2 = [...Array(NO)].map(e => Array(NH).fill(0.0));
let dW2 = [...Array(NO)].map(e => Array(NH).fill(0.0));

//Biases 1 and 2
let B1 = [...Array(NH).fill(0.0)];
let B2 = [...Array(NO).fill(0.0)];

let I = [...Array(NI).fill(0.0)];

let Z1 = [...Array(NH).fill(0.0)];
let Z2 = [...Array(NH).fill(0.0)];
let H = [...Array(NO).fill(0.0)];
let O = [...Array(NO).fill(0.0)];

//Randomising W1 and W2
function randomise (){
    B1 = Array.from({length: NH}, () => Math.random());
    B2 = Array.from({length: NO}, () => Math.random());

    for (let i = 0; i < NH; i++)
        W1[i] = Array.from({length: NI}, () => Math.random() * 0.3);

    for (let i = 0; i < NO; i++)
        W2[i] = Array.from({length: NH}, () => Math.random() * 0.3);
}

function forward (input){
    I = [].concat(input);
    for (let i = 0; i < NH; i++) {
        let weight = B1[i];
        for (let j = 0; j < NI; j++) 
            weight += W1[i][j] * I[j];
        Z2[i] = f(weight);
        Z1[i] = weight; //Needed to adjust the weight changes in back prop
    }
    for (let i = 0; i < NO; i++) {
        let weight = B2[i];
        for (let j = 0; j < NH; j++) 
            weight += W2[i][j] * Z2[j];
        O[i] = f(weight);
        H[i] = weight; //Needed to adjust the weight changes in back prop
    }
    return O;
}

function backward(output){
    //Computing deltas for output and hidden layer
    let D2 =  [...Array(NO).fill(0.0)];
    let error = 0.0;
    for (let i = 0; i < NO; i++) {
        D2[i] = (output[i] - O[i]) * derivative(H[i]);
        for (let j = 0; j < NH; j++) {
            dW2[i][j] += D2[i] * Z2[j]; //The weight changes = change in weight 
        }
    }

    //Computing deltas for output and hidden layer
    let D1 = [...Array(NH).fill(0.0)];
    for (let i = 0; i < NH; i++) {
        for (let j = 0; j < NO; j++) {
            D1[i] += D2[j] * W2[j][i];
        }
        D1[i] *= derivative(Z1[i]);

        for (let k = 0; k < NI; k++) {
            dW1[i][k] += D1[i] * I[k];
        }
    }

    for (let i = 0; i < NO; i++)
        error += Math.pow(output[i] - O[i], 2);
    return error * 0.5;
}

function updateWeights(learningValue) {
    for (let i = 0; i < NH; i++) 
        for (let j = 0; j < NI; j++) 
            W1[i][j] += learningValue * dW1[i][j];
    
    for (let i = 0; i < NO; i++) 
        for (let j = 0; j < NH; j++) 
            W2[i][j] += learningValue * dW2[i][j];

    dW1 = [...Array(NH)].map(e => Array(NI).fill(0.0));
    dW2 = [...Array(NO)].map(e => Array(NH).fill(0.0));
}

function f(x) {
    if (tan) return Math.tanh(x);
    return 1.0 / (1.0 + Math.exp(-x)); //Sigmoid
}

function derivative(x) {
    if (tan) return (1.0 - (Math.pow(f(x), 2)));
    return f(x) * (1.0 - f(x)); //Sigmoid
}
function question1 (){
    let inputs = [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0]
    ]
    let numExamples = inputs.length;
    
    let outputs = [
        [0.0],
        [1.0],
        [1.0],
        [0.0]
    ];

    randomise();
    for (let e = 0; e < maxEpochs; e++){
        error = 0.0;
        for (let p = 0; p < numExamples; p++){
            forward(inputs[p]);
            error += backward(outputs[p]);
            if (Math.random() < updateFreq)
                updateWeights(learningValue); //Small value for learning
        }
        console.log("Error at " + e + " is " + error);
    }

    let testingError = (Math.pow((forward(inputs[0]) - outputs[0]), 2) * 0.5) + (Math.pow((forward(inputs[1]) - outputs[1]), 2) * 0.5) + (Math.pow((forward(inputs[2]) - outputs[2]), 2) * 0.5) + (Math.pow((forward(inputs[3]) - outputs[3]), 2) * 0.5);
        
    //Printing the results
    process.stdout.write("\n  Results:\n|");
    for (let i = 0; i < outputs.length * 2; i++){
        process.stdout.write("--------");
    }
    process.stdout.write("|\n| ");
    for (let i = 0; i < outputs.length; i++){
        process.stdout.write(outputs[i][0] + "\t\t | ");
    }
    process.stdout.write("\n|");
    for (let i = 0; i < outputs.length * 2; i++){
        process.stdout.write("--------");
    }
    process.stdout.write("|\n| ");
    for (let i = 0; i < inputs.length; i++){
        let outcome = forward(inputs[i]);
        process.stdout.write((Math.round(outcome[0]* 10000) / 10000) + "\t | ");
    }
    process.stdout.write("\n|");
    for (let i = 0; i < outputs.length * 2; i++){
        process.stdout.write("--------");
    }
    process.stdout.write("|\n");

    console.log("Total testing error --> " + testingError);
}

function generateInputs (numInputs){
    let array = [...Array(numInputs)].map(e => Array(4).fill(0.0));
    for (let i = 0; i < numInputs; i++)
        array[i] = Array.from({length: 4}, () => (Math.random() * (1 - (-1)) + (-1)).toFixed(5));
    return array;
}

function generateOutputs (numInputs, inputs){
    let array = [...Array(numInputs)].map(e => Array(1).fill(0.0));
    //console.log (array);
    for (let i = 0; i < numInputs; i++){
        array[i] = Array.from({length: 1}, () => (Math.sin(inputs[i][0] - inputs[i][1] + parseInt(inputs[i][2]) - inputs[i][3])));
    }
    return array;
}

function question2 (){
    let inputsAll = generateInputs(500);
    let outputsAll = generateOutputs(500, inputsAll);

    //Split into 400 input and 100 testing
    let trainingInputs = inputsAll.slice(0, 400);
    let trainingOutputs = outputsAll.slice(0, 400);
    let testingInputs = inputsAll.slice(400, 500);
    let testingOutputs = outputsAll.slice(400, 500);

    //console.log(trainingOutputs);
    let error = 0.0;
    randomise();
    for (let e = 0; e < maxEpochs; e++){
        error = 0.0;
        for (let p = 0; p < trainingInputs.length; p++){
            forward(trainingInputs[p]);
            error += backward(trainingOutputs[p]);
            if (Math.random() < updateFreq){
                updateWeights(learningValue);
            }
        }
        console.log("Error at " + e + " is " + error);
    }

    let testingError = 0;
    process.stdout.write("\n|----------------------------------|\n   Can the MLP correctly train?\n|----------------------------------|\n    Testing Val     True Val\n|----------------------------------|\n");
    for (let i = 0; i < testingInputs.length; i++){
        process.stdout.write("\t");
        let test = [...Array(NO).fill(0.0)];
        test = forward(testingInputs[i]);
        testingError += (Math.pow(testingInputs[i][0] - test[0], 2) * 0.5);
        console.log((Math.round(test[0] * 1000) / 1000) + " \t--->\t " + (Math.round(testingOutputs[i][0] * 1000) / 1000) + "\n|----------------------------------|");
    }
    console.log("Total testing error --> " + testingError);
}

function indexofMax(arr) {
    var max = 0.0;
    var maxIndex = 0;

    for (var i = 0; i < arr.length; i++) {
        if (arr[i] > max) {
            maxIndex = i;
            max = arr[i];
        }
    }
    return maxIndex;
}

function question3 (){
    var fs = require('fs'); 

    let text = fs.readFileSync("letter-recognition.data").toString();
    text = text.replace(/\n/g, ",");

    let result = text.match(/(?:[^,]+(?:,|$)){1,17}/g);
    //result = result.slice(0, 500);
    
    let inputs = [...Array(result.length)].map(e => Array(16).fill(0.0));
    let outputs = [...Array(result.length)].map(e => Array(26).fill(0.0));

    for (let i = 0; i < result.length; i++){
        let temp = result[i].split(",");
        outputs[i][temp[0].charCodeAt(0) - 65] = 1.0;
        inputs[i] = temp.slice(1, 17);
    }

    let proportion = .75;
    let trainingInputs = inputs.slice(0, result.length * proportion);
    let trainingOutputs = outputs.slice(0, result.length * proportion);
    let testingInputs = inputs.slice(result.length * proportion, result.length);
    let testingOutputs = outputs.slice(result.length * proportion, result.length);

    randomise();
    fs.writeFile('file.txt', "Training Results\n", (err) => {
        if (err) throw err;
    })
    for (let e = 0; e < maxEpochs; e++) {
        error = 0.0;
        for (let p = 0;  p < trainingInputs.length; p++) {
            forward(trainingInputs[p]);
            error += backward(trainingOutputs[p]);
            if (Math.random() < updateFreq)
                updateWeights(learningValue);
        }
        console.log("Error at " + e + " is " + error);
        //Takes a lot of time to run, printing to the file 
        fs.appendFile('file.txt', e.toString() + ": " + error.toString() + "\n", 'utf8', (err) => { })
    }

    console.log("\n\nTESTING");
    let testingResults = [...Array(testingInputs.length)].map(e  => Array(4).fill(0.0));
    let testingError = 0, totalTestingError = 0
    for (let i = 0; i < testingInputs.length; i++){
        testingError = 0 
        let test = [...Array(NO).fill(0.0)];
        test = forward(testingInputs[i]);
        for (let j = 0; j < test.length; j++){
            testingError += (Math.pow(testingOutputs[i][0] - test[0], 2) * 0.5);
        }
        //The array for printing: index of element of max value of actual output, index of element of max value of predicted output, individual test error
        testingResults[i] = [indexofMax(testingOutputs[i]), indexofMax(test), testingError];
        totalTestingError += testingError;
    }
    console.log("Total testing error --> " + totalTestingError);

    //Printing the information
    fs.appendFile('file.txt', "Testing Results\nActual Answer\tPredicted Answer\tIndiv. Error\t\tCorrect?\t\n", (err) => { });

    process.stdout.write("Actual Answer\tPredicted Answer\tIndiv. Error\t\tCorrect?\t\n");
    let count = 0;
    for (let i = 0; i < testingResults.length; i++){
        if (testingResults[i][0] == testingResults[i][1]){
            testingResults[i][3] = 89;
            count ++;
        } else {
            testingResults[i][3] = 78;
        }
        //Adding 66 To make up for 0th elements
        process.stdout.write(String.fromCharCode(testingResults[i][0] + 65) + "\t\t" + String.fromCharCode(testingResults[i][1] + 65) + "\t\t" + testingResults[i][2] + "\t\t\t" + String.fromCharCode(testingResults[i][3]) +  "\t\n");
        fs.appendFile('file.txt', String.fromCharCode(testingResults[i][0] + 65) + "\t\t" + String.fromCharCode(testingResults[i][1] + 65) + "\t\t" + testingResults[i][2] + "\t\t\t" + String.fromCharCode(testingResults[i][3]) +  "\t\n", (err) => { });
    }
    console.log("No. correct answers: " + count + " No tests: " + testingResults.length + "\n==>Accuracy: " + ((count/testingResults.length)*100) + "%");
    fs.appendFile('file.txt', "No. correct answers: " + count + " No tests: " + testingResults.length + "\n==>Accuracy: " + ((count/testingResults.length)*100) + "%", (err) => { });
}

function main (){
    if (questionNumber == 1){
        question1();
    } else if (questionNumber == 2){
        question2();
    } else if (questionNumber == 3){
        question3();
    }
}

main();