const axios = require('axios');
const { performance } = require('perf_hooks');

const BATCHER_URL = 'http://localhost:3000/v1/chat/completions';
const NUM_USERS = 50;
const NUM_REQUESTS_PER_USER = 1; // Number of requests each user will make

// Function to make a single request
async function makeRequest(userId) {
    const startTime = performance.now();
    try {
        const response = await axios.post(BATCHER_URL, {
            model: 'nvidia/Llama-3.1-Nemotron-Nano-8B-v1',
            messages: [
                {
                    role: 'user',
                    content: `Hello from user ${userId}`
                }
            ],
            stream: false
        });

        const endTime = performance.now();
        const duration = endTime - startTime;

        return {
            success: true,
            duration,
            userId,
            responseTime: response.headers['x-response-time'] || 'N/A'
        };
    } catch (error) {
        const endTime = performance.now();
        const duration = endTime - startTime;

        return {
            success: false,
            duration,
            userId,
            error: error.message
        };
    }
}

// Function to simulate multiple users
async function simulateUsers() {
    console.log(`Starting benchmark with ${NUM_USERS} concurrent users...`);
    const startTime = performance.now();

    // Create an array of promises for all user requests
    const userPromises = Array.from({ length: NUM_USERS }, (_, i) => 
        Array.from({ length: NUM_REQUESTS_PER_USER }, () => makeRequest(i + 1))
    ).flat();

    // Wait for all requests to complete
    const results = await Promise.all(userPromises);

    const endTime = performance.now();
    const totalDuration = endTime - startTime;

    // Calculate statistics
    const successfulRequests = results.filter(r => r.success);
    const failedRequests = results.filter(r => !r.success);
    const averageDuration = successfulRequests.reduce((acc, curr) => acc + curr.duration, 0) / successfulRequests.length;
    const minDuration = Math.min(...successfulRequests.map(r => r.duration));
    const maxDuration = Math.max(...successfulRequests.map(r => r.duration));

    // Print results
    console.log('\nBenchmark Results:');
    console.log('------------------');
    console.log(`Total Requests: ${results.length}`);
    console.log(`Successful Requests: ${successfulRequests.length}`);
    console.log(`Failed Requests: ${failedRequests.length}`);
    console.log(`Total Duration: ${totalDuration.toFixed(2)}ms`);
    console.log(`Average Request Duration: ${averageDuration.toFixed(2)}ms`);
    console.log(`Min Request Duration: ${minDuration.toFixed(2)}ms`);
    console.log(`Max Request Duration: ${maxDuration.toFixed(2)}ms`);
    console.log(`Requests per Second: ${(results.length / (totalDuration / 1000)).toFixed(2)}`);

    if (failedRequests.length > 0) {
        console.log('\nFailed Requests:');
        failedRequests.forEach(req => {
            console.log(`User ${req.userId}: ${req.error}`);
        });
    }
}

// Run the benchmark
simulateUsers().catch(console.error); 