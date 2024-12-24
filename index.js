// erp_optimizer.js
const { AzureOpenAI } = require('openai');
const { Command } = require('commander');
const fs = require('fs');
const readline = require('readline');
const axios = require('axios');
const winston = require('winston');
const math = require('mathjs');
require('dotenv').config();

const openai = new AzureOpenAI({ endpoint: process.env.AZURE_OPENAI_ENDPOINT, apiKey: process.env.API_KEY, apiVersion: process.env.API_VERSION });

// 配置日志记录
const logger = winston.createLogger({
    level: 'info',
    format: winston.format.combine(
        winston.format.colorize(),
        winston.format.simple()
    ),
    transports: [new winston.transports.Console()],
});

/**
 * 实现Softmax函数，用于将评分转换为选择概率。
 * @param {number[]} scores - 评分数组
 * @param {number} temperature - 温度参数，控制随机性
 * @returns {number[]} - 选择概率数组
 */
function softmax(scores, temperature = 0.5) {
    const scaledScores = scores.map(score => score / temperature);
    const expScores = scaledScores.map(score => Math.exp(score));
    const sumExp = expScores.reduce((a, b) => a + b, 0);
    return expScores.map(exp => exp / sumExp);
}

// 类定义

class Feedback {
    constructor(content, score = 0.8) {
        this.content = content;
        this.score = score;
    }
}

/**
 * 示例数据类。
 */
class Exemplar {
    constructor(text, label, solution, score = 0.8) {
        this.text = text;
        this.label = label;
        this.solution = solution;
        this.score = score;
    }
}

/**
 * 基础内存管理类，提供评分更新和概率计算的方法。
 */
class MemoryBase {
    constructor(temperature = 0.5, threshold = 0.3, beta = 0.2) {
        this.temperature = temperature;
        this.threshold = threshold;
        this.beta = beta;
    }

    /**
     * 根据性能更新评分。
     * @param {number} score - 当前评分
     * @param {boolean} improved - 是否有性能提升
     * @param {number} performanceGain - 性能增益比例
     * @returns {number} - 更新后的评分
     */
    updateScore(score, improved, performanceGain) {
        if (improved) {
            const gain = Math.max(0.0, Math.min(1.0, performanceGain));
            score = (1 - this.beta) * score + this.beta * (1.0 + gain);
        } else {
            const penalty = Math.max(0.0, Math.min(1.0, Math.abs(performanceGain)));
            score = (1 - this.beta) * score - this.beta * penalty;
        }
        return Math.max(0.0, Math.min(1.0, score));
    }

    /**
     * 使用Softmax将评分转换为选择概率。
     * @param {number[]} scores - 评分数组
     * @returns {number[]} - 选择概率数组
     */
    getSelectionProbs(scores) {
        return softmax(scores, this.temperature);
    }
}

/**
 * 反馈内存类，继承自MemoryBase，用于管理反馈数据。
 */
class FeedbackMemory extends MemoryBase {
    constructor(temperature, threshold, beta) {
        super(temperature, threshold, beta);
        this.feedbacks = [];
    }

    /**
     * 添加新的反馈到内存。
     * @param {string} feedback - 反馈内容
     */
    addFeedback(feedback) {
        this.feedbacks.push(new Feedback(feedback));
    }

    /**
     * 根据评分检索前n个反馈。
     * @param {number} n - 要检索的反馈数量
     * @returns {string[]} - 反馈内容数组
     */
    retrieveFeedbacks(n = 3) {
        if (this.feedbacks.length === 0) return [];

        const scores = this.feedbacks.map(fb => fb.score);
        const probs = this.getSelectionProbs(scores);

        const indices = sampleIndices(this.feedbacks.length, Math.min(n, this.feedbacks.length), probs);
        return indices.map(i => this.feedbacks[i].content);
    }

    /**
     * 根据性能更新反馈评分，并移除评分低于阈值的反馈。
     * @param {string[]} feedbacks - 要更新的反馈内容数组
     * @param {boolean} improved - 是否有性能提升
     * @param {number} performanceGain - 性能增益比例
     */
    updateScores(feedbacks, improved, performanceGain = 0.0) {
        for (let fb of this.feedbacks) {
            if (feedbacks.includes(fb.content)) {
                fb.score = this.updateScore(fb.score, improved, performanceGain);
            }
        }
        this.feedbacks = this.feedbacks.filter(fb => fb.score >= this.threshold);
    }
}

/**
 * 示例工厂类，继承自MemoryBase，用于管理示例数据。
 */
class ExemplarFactory extends MemoryBase {
    constructor(temperature, threshold, beta) {
        super(temperature, threshold, beta);
        this.exemplars = [];
    }

    /**
     * 添加新的示例到工厂。
     * @param {string} text - 示例文本
     * @param {string} label - 示例标签
     * @param {string} solution - 解决方案
     */
    addExemplar(text, label, solution) {
        this.exemplars.push(new Exemplar(text, label, solution));
    }

    /**
     * 根据评分检索前n个示例。
     * @param {number} n - 要检索的示例数量
     * @returns {Object[]} - 示例对象数组
     */
    retrieveExemplars(n = 5) {
        if (this.exemplars.length === 0) return [];

        const scores = this.exemplars.map(ex => ex.score);
        const probs = this.getSelectionProbs(scores);

        const indices = sampleIndices(this.exemplars.length, Math.min(n, this.exemplars.length), probs);
        return indices.map(i => ({
            text: this.exemplars[i].text,
            label: this.exemplars[i].label,
            solution: this.exemplars[i].solution,
        }));
    }

    /**
     * 根据性能更新示例评分，并移除评分低于阈值的示例。
     * @param {Object[]} exemplars - 要更新的示例对象数组
     * @param {boolean} improved - 是否有性能提升
     * @param {number} performanceGain - 性能增益比例
     */
    updateScores(exemplars, improved, performanceGain = 0.0) {
        for (let ex of this.exemplars) {
            if (exemplars.some(e => e.text === ex.text && e.label === ex.label && e.solution === ex.solution)) {
                ex.score = this.updateScore(ex.score, improved, performanceGain);
            }
        }
        this.exemplars = this.exemplars.filter(ex => ex.score >= this.threshold);
    }
}

// Helper function to sample indices based on probabilities
function sampleIndices(length, n, probs) {
    const indices = [];
    const cumulativeProbs = probs.slice();
    for (let i = 1; i < cumulativeProbs.length; i++) {
        cumulativeProbs[i] += cumulativeProbs[i - 1];
    }

    for (let i = 0; i < n; i++) {
        const rand = Math.random();
        for (let j = 0; j < cumulativeProbs.length; j++) {
            if (rand < cumulativeProbs[j]) {
                if (!indices.includes(j)) {
                    indices.push(j);
                    break;
                }
            }
        }
    }
    return indices;
}

// Response Models (using simple objects)

class ExemplarResponse {
    constructor(text, label, solution) {
        this.text = text;
        this.label = label;
        this.solution = solution;
    }
}

class FeedbackResponse {
    constructor(feedback) {
        this.feedback = feedback;
    }
}

async function gen(modelSize, prompt, json_mode = false) {

    try {
        const options = {
            model: modelSize=='large' ? process.env.MODEL_NAME : process.env.SMALL_MODEL_NAME,
            messages: [{ role: 'system', content: prompt }],
        }
        if(json_mode){
            options.response_format = { type: 'json_object' }
        }
        const response =  await openai.chat.completions.create(options);
        return response.choices[0].message.content
    } catch (error) {
        logger.error(`Error in gen: ${error}`);
        return '';
    }
}

// Function to accumulate prompt
class PromptBuilder {
    constructor() {
        this.parts = [];
    }

    grow(text) {
        this.parts.push(text);
    }

    wrap(tag, content) {
        return `<${tag}>${content}</${tag}>`;
    }

    toString() {
        return this.parts.join('\n');
    }
}

// Functions

async function getExemplarsAndFeedback(prompt, errorSamples, numExemplars = 4, numFeedbacks = 3) {
    const builder = new PromptBuilder();
    builder.grow("I'm trying to write and complete a zero-shot classifier prompt from difficult or erroneous examples, 'text' field means model input, 'label' field means true label.");
    builder.grow(`The current prompt is:\n${prompt}`);
    builder.grow("But this prompt gets the following examples wrong:");
    for (let sample of errorSamples) {
        builder.grow(`text: ${sample.input}`);
        builder.grow(`label: ${sample.target}`);
    }
    builder.grow(`To improve my understanding and performance, I would like to identify ${numExemplars} typical examples from the above cases where the current prompt fails.`);
    builder.grow("These examples should be diverse to cover a range of different issues.");
    builder.grow("For each example, provide the following format in JSON and wrap each example with <key_example> and </key_example>:");
    builder.grow("{");
    builder.grow('"text": "{{input}}",');
    builder.grow('"label": "{{label}}",');
    builder.grow('"solution": "How to solve this problem step-by-step to get a more accurate answer."');
    builder.grow("}");

    builder.grow(`After identifying these ${numExemplars} typical examples, please provide ${numFeedbacks} reasons why the prompt could have gotten these examples wrong. Wrap each reason with <feedback> and </feedback>.`);

    const fullPrompt = builder.toString();

    const response = await gen('large', fullPrompt);

    // 解析响应中的示例和反馈
    const exemplars = [];
    const feedbacks = [];

    // 提取示例
    const exemplarRegex = /<key_example>([\s\S]*?)<\/key_example>/g;
    let match;
    while ((match = exemplarRegex.exec(response)) !== null) {
        try {
            const exemplarDict = JSON.parse(match[1]);
            exemplars.push(new ExemplarResponse(exemplarDict.text, exemplarDict.label, exemplarDict.solution));
        } catch (e) {
            logger.warn(`Failed to parse exemplar: ${e.message}`);
            continue;
        }
    }

    // 提取反馈
    const feedbackRegex = /<feedback>([\s\S]*?)<\/feedback>/g;
    while ((match = feedbackRegex.exec(response)) !== null) {
        feedbacks.push(new FeedbackResponse(match[1].trim()));
    }

    return [exemplars, feedbacks];
}

/**
 * 优化提示，基于反馈生成新的提示。
 * @param {string} prompt - 当前提示
 * @param {Object[]} errorSamples - 错误样本数组
 * @param {string[]} feedbacks - 反馈内容数组
 * @returns {Promise<string>} - 优化后的提示
 */
async function optimizePrompt(prompt, errorSamples, feedbacks) {
    const builder = new PromptBuilder();
    builder.grow("I'm trying to write and complete a zero-shot classifier prompt from difficult or erroneous examples, 'text' field means model input, 'label' field means true label.");
    builder.grow(`The current prompt is: ${prompt}`);
    builder.grow("But this prompt gets the following examples wrong:");
    for (let sample of errorSamples) {
        builder.grow(`Text: ${sample.input}`);
        builder.grow(`Label: ${sample.target}`);
    }
    builder.grow("Based on these examples the problem with this prompt is that:");
    for (let fb of feedbacks) {
        builder.grow(fb);
    }
    builder.grow("Based on the above information, I refine the prompt to make the model predict correctly.");
    builder.grow("The refined prompt is wrapped with <prompt> and </prompt>, less than 512 words:");

    const fullPrompt = builder.toString();

    let response = '';
    for (let i = 0; i < 5; i++) {
        try {
            response = await gen('large', fullPrompt);
            // 提取优化后的提示
            const match = response.match(/<prompt>([\s\S]*?)<\/prompt>/);
            if (match) {
                return match[1].trim();
            } else {
                if (i === 0) {
                    builder.grow("Please follow the format exactly.");
                } else {
                    builder.grow("You have failed on the <prompt> syntax, please try again.");
                }
            }
        } catch (error) {
            logger.error(`Error in optimizePrompt: ${error.message}`);
            if (i === 0) {
                builder.grow("The prompt can only contain `{input}` as the only variable.");
            } else {
                builder.grow("You have failed on including extra variables, please try again.");
            }
        }
    }

    return prompt;
}

async function loadData(dataPath) {
    const data = [];
    const fileStream = fs.createReadStream(dataPath);

    const rl = readline.createInterface({
        input: fileStream,
        crlfDelay: Infinity,
    });

    for await (const line of rl) {
        try {
            data.push(JSON.parse(line));
        } catch (e) {
            logger.warn(`Failed to parse line in ${dataPath}: ${e}`);
        }
    }

    return data;
}

function loadText(filePath) {
    return fs.readFileSync(filePath, 'utf-8').trim();
}

// ERM Class

class ERM {
    constructor(feedbackMemory = null, exemplarFactory = null, improvedPromptPath = null) {
        this.feedbackMemory = feedbackMemory || new FeedbackMemory();
        this.exemplarFactory = exemplarFactory || new ExemplarFactory();
        // 读取评估prompt
        this.evaluationStrict = !improvedPromptPath
        if(!this.evaluationStrict){
            this.evaluationPrompt = fs.readFileSync(improvedPromptPath, 'utf-8').trim();
        }
    }

    async evaluatePrompt(prompt, input, output = '') {

        const isTarget = (response, target) => {
            // 简单的匹配：检查目标标签是否在响应的最后一行中
            const lines = response.split('\n').map(line => line.trim().toLowerCase());
            return lines[lines.length - 1].includes(target.toLowerCase());
        };
        // 首先获取原始响应
        const filledPrompt = prompt.replace('{input}', input);
        const response = await gen('small', filledPrompt);
        if(this.evaluationStrict){
            return {
                response: response,
                score: isTarget(response, output) ? 1 : 0
            }
        }else {
        // 构建评估prompt
        const evaluationFilledPrompt = this.evaluationPrompt
            .replaceAll('{{input}}', input)
            .replaceAll('{{output}}', response)
            .replaceAll('{{reference_output}}', output);
            
        // 获取评估结果
        const evaluationResponse = await gen('large', evaluationFilledPrompt, true);
        console.log("evaluationResponse", response,output,evaluationResponse);
        try {
            const result = JSON.parse(evaluationResponse);
            // 返回原始响应和评分结果
            return {
                response: response,
                score: result.Score / (result?.FullScore) // 归一化分数
            };
        } catch (e) {
            logger.warn(`Failed to parse evaluation response: ${e}`);
            return {
                response: response,
                score: 0.5 // 默认中等分数
            };
        }
        }
    }

    async evaluate(prompt, data) {
        const results = [];
        for (let sample of data) {
            
            const evaluation = await this.evaluatePrompt(prompt, sample.input, sample.target);
            // 使用阈值0.75来判断是否为正确答案
            console.log("evaluation",evaluation, sample.target);
            const isCorrect = evaluation.score >= 0.75;
            results.push({
                correct: isCorrect,
                score: evaluation.score
            });
        }
        return results;
    }

    getErrorSamples(data, results, maxSamples = 5) {
        // 按照分数排序，选择得分最低的样本
        const samplesWithScores = data.map((sample, index) => ({
            sample: sample,
            score: results[index].score
        }));
        
        return samplesWithScores
            .sort((a, b) => a.score - b.score)
            .slice(0, maxSamples)
            .map(item => item.sample);
    }

    async optimize(initialPrompt, trainData, testData, numSteps = 10) {
        let bestPrompt = initialPrompt;
        let currentPrompt = initialPrompt;
        const trainResults = await this.evaluate(currentPrompt, trainData);
        let bestScore = trainResults.reduce((a, b) => a + b.score, 0) / trainResults.length;
        logger.info(`Initial score: ${bestScore}`);

        for (let step = 0; step < numSteps; step++) {
            // 获取分类错误的样本
            const errorSamples = this.getErrorSamples(trainData, trainResults);
            if (errorSamples.length === 0) {
                logger.info("No errors found, optimization complete");
                break;
            }

            // 获取示例和反馈
            const [exemplars, feedbacks] = await getExemplarsAndFeedback(currentPrompt, errorSamples);

            // Store exemplars and feedback
            for (let ex of exemplars) {
                this.exemplarFactory.addExemplar(ex.text, ex.label, ex.solution);
            }
            for (let fb of feedbacks) {
                this.feedbackMemory.addFeedback(fb.feedback);
            }

            // 检索存储的反馈
            const storedFeedbacks = this.feedbackMemory.retrieveFeedbacks();

            // 生成新的提示
            currentPrompt = await optimizePrompt(currentPrompt, errorSamples, storedFeedbacks);
            // console.log("\n\ncurrentPrompt", currentPrompt,'\n\n');
            // Evaluate on test data
            const testResults = await this.evaluate(currentPrompt, testData);
            const currentScore = testResults.reduce((a, b) => a + b.score, 0) / testResults.length;
            logger.info(`Step ${step + 1}, Score: ${currentScore}`);

            // 计算性能增益
            const performanceGain = (currentScore - bestScore) / Math.max(1e-6, bestScore);
            const improved = currentScore > bestScore;

            // 根据性能增益更新内存评分
            this.feedbackMemory.updateScores(storedFeedbacks, improved, performanceGain);
            this.exemplarFactory.updateScores(exemplars, improved, performanceGain);

            if (improved) {
                bestScore = currentScore;
                bestPrompt = currentPrompt;
                console.log("bestPrompt", bestPrompt);
            }
        }

        return [bestScore, bestPrompt];
    }
}

// 主函数

async function main() {
    const program = new Command();
    program
        .option('--dataset <type>', 'Dataset name', 'data')
        .option('--data-path <type>', 'Data path', '.')
        .option('--num-steps <number>', 'Number of optimization steps', '5')
        .option('--num-train-samples <number>', 'Number of training samples', null)
        .option('--num-val-samples <number>', 'Number of validation samples', null)
        .option('--num-test-samples <number>', 'Number of test samples', null)
        .option('--seed <number>', 'Random seed', '42');

    program.parse(process.argv);
    const options = program.opts();

    // 设置随机种子
    const seed = parseInt(options.seed);
    if (!isNaN(seed)) {
        Math.seedrandom(seed);
    }

    // 加载数据
    const trainDataPath = `${options.dataPath}/${options.dataset}/train.jsonl`;
    const valDataPath = `${options.dataPath}/${options.dataset}/eval.jsonl`;
    const testDataPath = `${options.dataPath}/${options.dataset}/test.jsonl`;
    const initialPromptPath = `${options.dataPath}/${options.dataset}/prompt.txt`;
    const improvedPromptPath = `${options.dataPath}/${options.dataset}/improved_prompt.txt`;

    const [trainData, valData, testData] = await Promise.all([
        loadData(trainDataPath),
        loadData(valDataPath),
        loadData(testDataPath),
    ]);

    let finalTrainData = trainData;
    let finalValData = valData;
    let finalTestData = testData;

    if (options.numTrainSamples) {
        finalTrainData = trainData.slice(0, parseInt(options.numTrainSamples));
    }
    if (options.numValSamples) {
        finalValData = valData.slice(0, parseInt(options.numValSamples));
    }
    if (options.numTestSamples) {
        finalTestData = testData.slice(0, parseInt(options.numTestSamples));
    }

    // 初始提示
    const initialPrompt = loadText(initialPromptPath).trim();

    // 初始化 ERM
    const erm = new ERM();

    // 运行优化
    const [finalScore, finalPrompt] = await erm.optimize(initialPrompt, finalTrainData, finalValData, parseInt(options.numSteps));

    logger.info(`Final score on validation set: ${finalScore}`);
    logger.info(`Final prompt:\n${finalPrompt}`);

    // 评估测试集
    const initTestResults = await erm.evaluate(initialPrompt, finalTestData);
    const initTestScore = initTestResults.reduce((a, b) => a + b.score, 0) / initTestResults.length;
    logger.info(`Initial score on test set: ${initTestScore}`);

    const finalTestResults = await erm.evaluate(finalPrompt, finalTestData);
    const finalTestScore = finalTestResults.reduce((a, b) => a + b.score, 0) / finalTestResults.length;
    logger.info(`Final score on test set: ${finalTestScore}`);

    // 保存最终提示
    const finalPromptPath = `${options.dataPath}/${options.dataset}/final_prompt.txt`;
    fs.writeFileSync(finalPromptPath, finalPrompt);
}

// 设置随机种子（需要安装 seedrandom 库）
const seedrandom = require('seedrandom');
Math.seedrandom = seedrandom;

// 执行主函数
if (require.main === module) {
    main().catch(err => {
        logger.error(`Error in main: ${err.message}`);
        process.exit(1);
    });
}
