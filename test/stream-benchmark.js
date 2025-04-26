const axios = require('axios');
const { performance } = require('perf_hooks');

const BATCHER_URL = 'http://localhost:3000/v1/chat/completions';
const NUM_USERS = 1000;
const NUM_REQUESTS_PER_USER = 1;

// Function to make a streaming request
async function makeStreamRequest(userId) {
    const startTime = performance.now();
    let chunksReceived = 0;
    let firstChunkTime = null;
    let lastChunkTime = null;

    try {
        const response = await axios.post(BATCHER_URL, {
            model: 'nvidia/Llama-3.1-Nemotron-Nano-8B-v1',
            messages: [
                {
                    role: 'user',
                    content: `Hello from user ${userId}`
                }
            ],
            stream: true
        }, {
            responseType: 'stream'
        });

        return new Promise((resolve, reject) => {
            response.data.on('data', (chunk) => {
                const lines = chunk.toString().split('\n').filter(line => line.trim() !== '');
                
                lines.forEach(line => {
                    if (line.startsWith('data: ')) {
                        const data = line.slice(6);
                        if (data === '[DONE]') {
                            const endTime = performance.now();
                            resolve({
                                success: true,
                                duration: endTime - startTime,
                                userId,
                                chunksReceived,
                                firstChunkTime: firstChunkTime ? firstChunkTime - startTime : null,
                                lastChunkTime: lastChunkTime ? lastChunkTime - startTime : null
                            });
                        } else {
                            chunksReceived++;
                            if (!firstChunkTime) {
                                firstChunkTime = performance.now();
                            }
                            lastChunkTime = performance.now();
                        }
                    }
                });
            });

            response.data.on('error', (error) => {
                const endTime = performance.now();
                reject({
                    success: false,
                    duration: endTime - startTime,
                    userId,
                    error: error.message
                });
            });
        });
    } catch (error) {
        const endTime = performance.now();
        return {
            success: false,
            duration: endTime - startTime,
            userId,
            error: error.message
        };
    }
}

// Function to simulate multiple streaming users
async function simulateStreamingUsers() {
    console.log(`Starting streaming benchmark with ${NUM_USERS} concurrent users...`);
    const startTime = performance.now();

    // Create an array of promises for all user requests
    const userPromises = Array.from({ length: NUM_USERS }, (_, i) => 
        Array.from({ length: NUM_REQUESTS_PER_USER }, () => makeStreamRequest(i + 1))
    ).flat();

    // Wait for all requests to complete
    const results = await Promise.all(userPromises);

    const endTime = performance.now();
    const totalDuration = endTime - startTime;

    // Calculate statistics
    const successfulRequests = results.filter(r => r.success);
    const failedRequests = results.filter(r => !r.success);
    
    const averageDuration = successfulRequests.reduce((acc, curr) => acc + curr.duration, 0) / successfulRequests.length;
    const averageFirstChunkTime = successfulRequests.reduce((acc, curr) => acc + (curr.firstChunkTime || 0), 0) / successfulRequests.length;
    const averageLastChunkTime = successfulRequests.reduce((acc, curr) => acc + (curr.lastChunkTime || 0), 0) / successfulRequests.length;
    
    const minDuration = Math.min(...successfulRequests.map(r => r.duration));
    const maxDuration = Math.max(...successfulRequests.map(r => r.duration));

    // Print results
    console.log('\nStreaming Benchmark Results:');
    console.log('----------------------------');
    console.log(`Total Requests: ${results.length}`);
    console.log(`Successful Requests: ${successfulRequests.length}`);
    console.log(`Failed Requests: ${failedRequests.length}`);
    console.log(`Total Duration: ${totalDuration.toFixed(2)}ms`);
    console.log(`Average Request Duration: ${averageDuration.toFixed(2)}ms`);
    console.log(`Average Time to First Chunk: ${averageFirstChunkTime.toFixed(2)}ms`);
    console.log(`Average Time to Last Chunk: ${averageLastChunkTime.toFixed(2)}ms`);
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
simulateStreamingUsers().catch(console.error); 