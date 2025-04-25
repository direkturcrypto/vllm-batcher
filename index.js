const express = require('express');
const bodyParser = require('body-parser');
const axios = require('axios');
const dotenv = require('dotenv');

dotenv.config();

const app = express();
const port = process.env.PORT || 3000;

// Configuration
const VLLM_ENDPOINT = `${process.env.VLLM_ENDPOINT || 'http://localhost:8080'}/v1/chat/completions`;
const MODEL = process.env.MODEL || 'unsloth/Llama-3.1-8B-Instruct-bnb-4bit';
const MAX_CONTEXT_LENGTH = process.env.MAX_CONTEXT_LENGTH || 8192;
const DEFAULT_MAX_TOKENS = 1024;
const BATCH_SIZE = 100;
const BATCH_INTERVAL = 100;
const MAX_QUEUE_SIZE = 1000;
const MAX_CONCURRENT_REQUESTS = 500;
const VLLM_STATUS_CHECK_INTERVAL = 10000;
const MAX_BATCH_TOKENS = process.env.MAX_BATCH_TOKENS || 100000; // Maximum tokens per batch
const MAX_VLLM_WAITING_REQUESTS = 5; // Maximum number of waiting requests for VLLM
const OPENROUTER_API_KEY = process.env.OPENROUTER_API_KEY;
const OPENROUTER_MODEL = process.env.OPENROUTER_MODEL || 'meta-llama/llama-3.1-8b-instruct';
const OPENROUTER_ENDPOINT = process.env.OPENROUTER_ENDPOINT || 'https://openrouter.ai/api/v1/chat/completions';

// Request queues and status
let requestQueue = [];
let streamRequestQueue = [];
let isProcessing = false;
let isStreamProcessing = false;
let activeRequests = 0;
let isVLLMAvailable = true;
let currentBatchTokens = 0;
let vllmWaitingRequests = 0; // Current count of waiting requests in VLLM

// Middleware
app.use(bodyParser.json());

// Function to normalize messages
function normalizeMessages(messages) {
    if (!Array.isArray(messages)) return messages;
    
    // First pass: collect and combine messages by role
    const roleMessages = {
        system: [],
        user: [],
        assistant: []
    };

    for (const message of messages) {
        if (!message.role || !message.content) continue;
        if (message.content.trim() === '') continue;

        // Force invalid roles to become 'user'
        const role = ['system', 'user', 'assistant'].includes(message.role) 
            ? message.role 
            : 'user';

        roleMessages[role].push(message.content.trim());
    }

    // Combine messages of the same role
    const combinedMessages = {
        system: roleMessages.system.join('\n'),
        user: roleMessages.user.join('\n'),
        assistant: roleMessages.assistant.join('\n')
    };

    // Build the final sequence
    const finalMessages = [];

    // Add system message if exists
    if (combinedMessages.system) {
        finalMessages.push({
            role: 'system',
            content: combinedMessages.system
        });
    }

    // Add user and assistant messages in alternating sequence
    const maxLength = Math.max(
        roleMessages.user.length,
        roleMessages.assistant.length
    );

    for (let i = 0; i < maxLength; i++) {
        // Add user message if exists
        if (roleMessages.user[i]) {
            finalMessages.push({
                role: 'user',
                content: roleMessages.user[i]
            });
        }

        // Add assistant message if exists
        if (roleMessages.assistant[i]) {
            finalMessages.push({
                role: 'assistant',
                content: roleMessages.assistant[i]
            });
        }
    }

    // If the last message is from user, add an empty assistant response
    if (finalMessages.length > 0 && finalMessages[finalMessages.length - 1].role === 'user') {
        finalMessages.push({
            role: 'assistant',
            content: ''
        });
    }

    return finalMessages;
}

// Function to estimate tokens in a message
function estimateTokens(messages) {
    if (!messages || !Array.isArray(messages)) return 0;
    
    // Simple estimation: ~4 chars per token for English text
    let totalChars = 0;
    for (const msg of messages) {
        if (msg.content) {
            totalChars += msg.content.length;
        }
    }
    
    return Math.ceil(totalChars / 4);
}

// Function to check VLLM server status and queue length
async function checkVLLMStatus() {
    try {
        // First check if the server is available at all
        const healthResponse = await axios.get(VLLM_ENDPOINT.replace('/v1/chat/completions', '/health'));
        
        if (healthResponse.status === 200) {
            isVLLMAvailable = true;
            
            // Try different methods to get queue statistics
            let foundStats = false;
            
            try {
                const metricsResponse = await axios.get(VLLM_ENDPOINT.replace('/v1/chat/completions', '/metrics'));
                if (metricsResponse.data) {
                    const waitingMatch = metricsResponse.data.toString().match(/waiting_requests\s+(\d+)/);
                    if (waitingMatch && waitingMatch[1]) {
                        vllmWaitingRequests = parseInt(waitingMatch[1], 10);
                        if (vllmWaitingRequests > 0) {
                            console.log(`VLLM waiting requests (from metrics): ${vllmWaitingRequests}`);
                            foundStats = true;
                        }
                    }
                }
            } catch (metricsError) {
                console.log('Could not fetch VLLM metrics:', metricsError.message);
            }
            
            // Method 3: If all else fails, base it on our actively sent requests
            if (!foundStats) {
                // Use our active requests as a rough estimate
                vllmWaitingRequests = Math.max(0, activeRequests - 5); // Assume at least 5 can be processed concurrently
                // console.log(`VLLM waiting requests (estimated): ${vllmWaitingRequests}`);
            }
        } else {
            isVLLMAvailable = false;
        }
    } catch (error) {
        isVLLMAvailable = false;
        console.error('VLLM server is not available:', error.message);
    }
}

// Function to normalize non-streaming response
function normalizeResponse(response, source = 'vllm') {
    if (!response) return null;

    // Return if it's already an error object
    if (response.error) return response;

    // Create a standard response format
    const normalizedResponse = {
        id: `chatcmpl-${Date.now()}-${Math.random().toString(36).substring(2, 10)}`,
        object: 'chat.completion',
        created: response.created || Math.floor(Date.now() / 1000),
        model: response.model || (source === 'vllm' ? MODEL : OPENROUTER_MODEL),
        choices: [],
        usage: response.usage || {
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0
        }
    };

    // Normalize choices
    if (response.choices && Array.isArray(response.choices)) {
        normalizedResponse.choices = response.choices.map(choice => {
            const normalizedChoice = {
                index: choice.index || 0,
                finish_reason: choice.finish_reason || 'stop'
            };

            // Handle different message formats
            if (choice.message) {
                normalizedChoice.message = {
                    role: choice.message.role || 'assistant',
                    content: choice.message.content || ''
                };
            } else if (choice.delta) {
                normalizedChoice.delta = {
                    role: choice.delta.role || 'assistant',
                    content: choice.delta.content || ''
                };
            } else if (typeof choice.text === 'string') {
                // Some models return just text
                normalizedChoice.message = {
                    role: 'assistant',
                    content: choice.text
                };
            }

            return normalizedChoice;
        });
    }

    return normalizedResponse;
}

// Function to normalize streaming chunk
function normalizeStreamChunk(chunk, source = 'vllm') {
    if (!chunk) return null;

    // If it's a string "[DONE]", return as is
    if (chunk === '[DONE]') return chunk;

    // If it's already an error object
    if (chunk.error) return chunk;

    try {
        // Parse chunk if it's a string
        const data = typeof chunk === 'string' ? JSON.parse(chunk) : chunk;

        // Create normalized chunk
        const normalizedChunk = {
            id: `chatcmpl-${Date.now()}-${Math.random().toString(36).substring(2, 10)}`,
            object: 'chat.completion.chunk',
            created: data.created || Math.floor(Date.now() / 1000),
            model: data.model || (source === 'vllm' ? MODEL : OPENROUTER_MODEL),
            choices: []
        };

        // Normalize choices
        if (data.choices && Array.isArray(data.choices)) {
            normalizedChunk.choices = data.choices.map(choice => {
                const normalizedChoice = {
                    index: choice.index || 0,
                    finish_reason: choice.finish_reason || null
                };

                // Handle different delta formats
                if (choice.delta) {
                    normalizedChoice.delta = {
                        role: choice.delta.role || undefined,
                        content: choice.delta.content || ''
                    };
                    
                    // Remove undefined fields
                    if (normalizedChoice.delta.role === undefined) {
                        delete normalizedChoice.delta.role;
                    }
                } else if (choice.message) {
                    normalizedChoice.delta = {
                        role: choice.message.role || 'assistant',
                        content: choice.message.content || ''
                    };
                } else if (typeof choice.text === 'string') {
                    normalizedChoice.delta = {
                        content: choice.text
                    };
                }

                return normalizedChoice;
            });
        }

        return normalizedChunk;
    } catch (error) {
        console.error('Error normalizing stream chunk:', error);
        return chunk; // Return original on error
    }
}

// Function to normalize error responses
function normalizeErrorResponse(error) {
    return {
        error: {
            message: typeof error === 'string' ? error : (error.message || 'Unknown error'),
            type: error.type || 'server_error',
            param: error.param,
            code: error.code
        }
    };
}

// Forward request to OpenRouter
async function forwardToOpenRouter(req, isStream = false) {
    try {
        const openRouterReq = {
            model: OPENROUTER_MODEL,
            messages: req.body.messages,
            stream: isStream,
            max_tokens: req.body.max_tokens || DEFAULT_MAX_TOKENS,
            temperature: req.body.temperature || 0.7,
            top_p: req.body.top_p || 1.0,
            n: req.body.n || 1,
            stop: req.body.stop,
            presence_penalty: req.body.presence_penalty || 0.0,
            frequency_penalty: req.body.frequency_penalty || 0.0,
            logit_bias: req.body.logit_bias,
            user: req.body.user
        };

        if (isStream) {
            // Handle streaming response
            const response = await axios.post(OPENROUTER_ENDPOINT, openRouterReq, {
                headers: {
                    'Authorization': `Bearer ${OPENROUTER_API_KEY}`,
                    'X-Title': 'vLLM Batcher'
                },
                responseType: 'stream'
            });
            
            return response.data;
        } else {
            // Handle non-streaming response
            const response = await axios.post(OPENROUTER_ENDPOINT, openRouterReq, {
                headers: {
                    'Authorization': `Bearer ${OPENROUTER_API_KEY}`,
                    'X-Title': 'vLLM Batcher'
                }
            });
            
            return normalizeResponse(response.data, 'openrouter');
        }
    } catch (error) {
        console.error('Error forwarding to OpenRouter:', error.message);
        // Create a standardized error object
        const errorResponse = {
            error: {
                message: error.response?.data?.error?.message || error.message || 'OpenRouter error',
                type: 'openrouter_error',
                code: error.response?.status || 500
            }
        };
        throw errorResponse; // Throw the standardized error
    }
}

// Add a logging function to track performance metrics
function logPerformanceMetrics() {
    if (vllmWaitingRequests > 0) {
        console.log(`=== PERFORMANCE METRICS ===`);
        console.log(`Active Requests: ${activeRequests}`);
        console.log(`Queue Size: ${requestQueue.length + streamRequestQueue.length} (${requestQueue.length} non-stream, ${streamRequestQueue.length} stream)`);
        console.log(`VLLM Waiting Requests: ${vllmWaitingRequests}`);
        console.log(`VLLM Available: ${isVLLMAvailable}`);
        console.log(`=========================`);
    }
}

// Start VLLM status checking and metrics logging
setInterval(checkVLLMStatus, VLLM_STATUS_CHECK_INTERVAL);
setInterval(logPerformanceMetrics, VLLM_STATUS_CHECK_INTERVAL);

// Function to process non-streaming batch
async function processBatch() {
    if (requestQueue.length === 0 || !isVLLMAvailable || activeRequests >= MAX_CONCURRENT_REQUESTS) return;

    // Reset token counter for new batch
    currentBatchTokens = 0;
    
    // Take as many requests as possible until reaching MAX_BATCH_TOKENS
    const batch = [];
    while (requestQueue.length > 0 && batch.length < BATCH_SIZE) {
        const req = requestQueue[0];
        const estimatedTokens = estimateTokens(req.body.messages) + (req.body.max_tokens || DEFAULT_MAX_TOKENS);
        
        // If this single request exceeds context length or we have too many waiting requests, forward to OpenRouter
        const tooManyWaitingRequests = vllmWaitingRequests >= MAX_VLLM_WAITING_REQUESTS;
        if (estimatedTokens > MAX_CONTEXT_LENGTH || tooManyWaitingRequests) {
            const req = requestQueue.shift();
            try {
                const reason = estimatedTokens > MAX_CONTEXT_LENGTH ? 'Context length exceeded' : 'Too many waiting requests';
                console.log(`Forwarding to OpenRouter from batch: ${reason}`);
                const openRouterResponse = await forwardToOpenRouter(req);
                req.resolve(openRouterResponse);
            } catch (error) {
                req.reject(error.response?.data || error.message);
            }
            continue;
        }
        
        // If adding this request would exceed batch token limit, break
        if (currentBatchTokens + estimatedTokens > MAX_BATCH_TOKENS) {
            break;
        }
        
        batch.push(requestQueue.shift());
        currentBatchTokens += estimatedTokens;
    }
    
    if (batch.length === 0) return;

    try {
        const requests = batch.map(req => {
            const normalizedMessages = normalizeMessages(req.body.messages);
            const maxTokens = Math.min(
                req.body.max_tokens || DEFAULT_MAX_TOKENS,
                MAX_CONTEXT_LENGTH - 1000
            );

            return {
                model: MODEL,
                messages: normalizedMessages,
                stream: false,
                max_tokens: maxTokens,
                temperature: req.body.temperature || 0.7,
                top_p: req.body.top_p || 1.0,
                n: req.body.n || 1,
                stop: req.body.stop,
                presence_penalty: req.body.presence_penalty || 0.0,
                frequency_penalty: req.body.frequency_penalty || 0.0,
                logit_bias: req.body.logit_bias,
                user: req.body.user
            };
        });

        activeRequests += requests.length;

        // Send all requests in parallel with immediate processing
        const responses = await Promise.all(
            requests.map(async (r, index) => {
                try {
                    const response = await axios.post(VLLM_ENDPOINT, r);
                    return { index, data: normalizeResponse(response.data, 'vllm') };
                } catch (error) {
                    // Standardize error format
                    const errorResponse = {
                        error: {
                            message: error.response?.data?.error?.message || error.message || 'VLLM processing error',
                            type: 'vllm_error',
                            code: error.response?.status || 500
                        }
                    };
                    return { index, error: errorResponse };
                } finally {
                    activeRequests--;
                }
            })
        );

        // Resolve/reject all promises in parallel
        await Promise.all(
            responses.map(({ index, data, error }) => {
                if (error) {
                    return batch[index].reject(error);
                } else {
                    return batch[index].resolve(data);
                }
            })
        );
    } catch (error) {
        await Promise.all(
            batch.map(req => {
                // Standardize error format
                const errorResponse = {
                    error: {
                        message: error.response?.data?.error?.message || error.message || 'Batch processing error',
                        type: 'batch_error',
                        code: error.response?.status || 500
                    }
                };
                req.reject(errorResponse);
            })
        );
    } finally {
        // Process next batch immediately if there are remaining requests
        if (requestQueue.length > 0) {
            processBatch();
        }
    }
}

// Function to process streaming batch
async function processStreamBatch() {
    if (streamRequestQueue.length === 0 || !isVLLMAvailable || activeRequests >= MAX_CONCURRENT_REQUESTS) return;

    // Reset token counter for new batch
    currentBatchTokens = 0;
    
    // Take as many requests as possible until reaching MAX_BATCH_TOKENS
    const batch = [];
    while (streamRequestQueue.length > 0 && batch.length < BATCH_SIZE) {
        const req = streamRequestQueue[0];
        const estimatedTokens = estimateTokens(req.body.messages) + (req.body.max_tokens || DEFAULT_MAX_TOKENS);
        
        // If this single request exceeds context length or we have too many waiting requests, forward to OpenRouter
        const tooManyWaitingRequests = vllmWaitingRequests >= MAX_VLLM_WAITING_REQUESTS;
        if (estimatedTokens > MAX_CONTEXT_LENGTH || tooManyWaitingRequests) {
            const req = streamRequestQueue.shift();
            try {
                const reason = estimatedTokens > MAX_CONTEXT_LENGTH ? 'Context length exceeded' : 'Too many waiting requests';
                console.log(`Forwarding to OpenRouter from stream batch: ${reason}`);
                const openRouterStream = await forwardToOpenRouter(req, true);
                const { res } = req;
                let isEnded = false;
                
                res.setHeader('Content-Type', 'text/event-stream');
                res.setHeader('Cache-Control', 'no-cache');
                res.setHeader('Connection', 'keep-alive');
                
                let buffer = '';
                openRouterStream.on('data', (chunk) => {
                    if (isEnded) return;
                    
                    buffer += chunk.toString();
                    const lines = buffer.split('\n');
                    buffer = lines.pop() || '';
                    
                    lines.forEach(line => {
                        if (line.trim() === '') return;
                        if (line.startsWith('data: ')) {
                            const data = line.slice(6);
                            if (data === '[DONE]') {
                                if (!isEnded) {
                                    res.write('data: [DONE]\n\n');
                                    res.end();
                                    isEnded = true;
                                }
                            } else {
                                try {
                                    const parsed = JSON.parse(data);
                                    const normalized = normalizeStreamChunk(parsed, 'openrouter');
                                    if (!isEnded) {
                                        res.write(`data: ${JSON.stringify(normalized)}\n\n`);
                                    }
                                } catch (e) {
                                    console.error('Error parsing stream data:', e);
                                }
                            }
                        }
                    });
                });
                
                openRouterStream.on('end', () => {
                    if (!isEnded) {
                        res.write('data: [DONE]\n\n');
                        res.end();
                        isEnded = true;
                    }
                });
                
                openRouterStream.on('error', (error) => {
                    if (!isEnded) {
                        res.write(`data: ${JSON.stringify({ error: error.message })}\n\n`);
                        res.end();
                        isEnded = true;
                    }
                });
                
                res.on('close', () => {
                    isEnded = true;
                });
            } catch (error) {
                const { res } = req;
                if (!res.writableEnded) {
                    res.write(`data: ${JSON.stringify({ error: error.message })}\n\n`);
                    res.end();
                }
            }
            continue;
        }
        
        // If adding this request would exceed batch token limit, break
        if (currentBatchTokens + estimatedTokens > MAX_BATCH_TOKENS) {
            break;
        }
        
        batch.push(streamRequestQueue.shift());
        currentBatchTokens += estimatedTokens;
    }
    
    if (batch.length === 0) return;

    try {
        const requests = batch.map(req => {
            const normalizedMessages = normalizeMessages(req.body.messages);
            const maxTokens = Math.min(
                req.body.max_tokens || DEFAULT_MAX_TOKENS,
                MAX_CONTEXT_LENGTH - 1000
            );

            return {
                model: MODEL,
                messages: normalizedMessages,
                stream: true,
                max_tokens: maxTokens,
                temperature: req.body.temperature || 0.7,
                top_p: req.body.top_p || 1.0,
                n: req.body.n || 1,
                stop: req.body.stop,
                presence_penalty: req.body.presence_penalty || 0.0,
                frequency_penalty: req.body.frequency_penalty || 0.0,
                logit_bias: req.body.logit_bias,
                user: req.body.user
            };
        });

        activeRequests += requests.length;

        // Send all requests in parallel with immediate processing
        const responses = await Promise.all(
            requests.map(async (r, index) => {
                try {
                    const response = await axios.post(VLLM_ENDPOINT, r, {
                        responseType: 'stream',
                        validateStatus: status => status >= 200 && status < 500
                    });

                    if (response.status >= 400) {
                        let errorData = '';
                        for await (const chunk of response.data) {
                            errorData += chunk.toString();
                        }
                        try {
                            const errorJson = JSON.parse(errorData);
                            return { index, error: errorJson };
                        } catch (e) {
                            return { index, error: errorData };
                        }
                    }

                    return { index, data: response.data };
                } catch (error) {
                    if (error.response) {
                        let errorData = '';
                        if (error.response.data) {
                            if (typeof error.response.data === 'string') {
                                errorData = error.response.data;
                            } else if (error.response.data.on) {
                                for await (const chunk of error.response.data) {
                                    errorData += chunk.toString();
                                }
                            } else {
                                errorData = JSON.stringify(error.response.data);
                            }
                        }
                        return { index, error: errorData || error.message };
                    } else if (error.request) {
                        return { index, error: 'No response received from server' };
                    } else {
                        return { index, error: error.message };
                    }
                } finally {
                    activeRequests--;
                }
            })
        );

        // Handle all responses in parallel
        await Promise.all(
            responses.map(({ index, data, error }) => {
                const { res } = batch[index];
                let isEnded = false;

                res.setHeader('Content-Type', 'text/event-stream');
                res.setHeader('Cache-Control', 'no-cache');
                res.setHeader('Connection', 'keep-alive');

                if (error) {
                    if (!isEnded) {
                        // Standardize error format
                        const errorResponse = {
                            error: {
                                message: typeof error === 'string' ? error : JSON.stringify(error),
                                type: 'vllm_error'
                            }
                        };
                        res.write(`data: ${JSON.stringify(errorResponse)}\n\n`);
                        res.end();
                        isEnded = true;
                    }
                    return Promise.resolve();
                }

                return new Promise((resolve) => {
                    let buffer = '';
                    data.on('data', (chunk) => {
                        if (isEnded) return;
                        
                        buffer += chunk.toString();
                        const lines = buffer.split('\n');
                        buffer = lines.pop() || '';

                        lines.forEach(line => {
                            if (line.trim() === '') return;
                            if (line.startsWith('data: ')) {
                                const data = line.slice(6);
                                if (data === '[DONE]') {
                                    if (!isEnded) {
                                        res.write('data: [DONE]\n\n');
                                        res.end();
                                        isEnded = true;
                                        resolve();
                                    }
                                } else {
                                    try {
                                        const parsed = JSON.parse(data);
                                        const normalized = normalizeStreamChunk(parsed, 'vllm');
                                        if (!isEnded) {
                                            res.write(`data: ${JSON.stringify(normalized)}\n\n`);
                                        }
                                    } catch (e) {
                                        console.error('Error parsing stream data:', e);
                                    }
                                }
                            }
                        });
                    });

                    data.on('end', () => {
                        if (isEnded) return;
                        
                        if (buffer.trim()) {
                            try {
                                const data = buffer.slice(6);
                                const parsed = JSON.parse(data);
                                const normalized = normalizeStreamChunk(parsed, 'vllm');
                                if (!isEnded) {
                                    res.write(`data: ${JSON.stringify(normalized)}\n\n`);
                                }
                            } catch (e) {
                                console.error('Error parsing final buffer:', e);
                            }
                        }
                        if (!isEnded) {
                            res.write('data: [DONE]\n\n');
                            res.end();
                            isEnded = true;
                        }
                        resolve();
                    });

                    data.on('error', (error) => {
                        if (!isEnded) {
                            // Standardize error format
                            const errorResponse = {
                                error: {
                                    message: error.message || 'Stream error',
                                    type: 'stream_error'
                                }
                            };
                            res.write(`data: ${JSON.stringify(errorResponse)}\n\n`);
                            res.end();
                            isEnded = true;
                        }
                        resolve();
                    });

                    res.on('close', () => {
                        isEnded = true;
                        resolve();
                    });
                });
            })
        );
    } catch (error) {
        await Promise.all(
            batch.map(req => {
                if (!req.res.writableEnded) {
                    // Standardize error format
                    const errorResponse = {
                        error: {
                            message: error.response?.data || error.message,
                            type: 'batch_processing_error'
                        }
                    };
                    req.res.write(`data: ${JSON.stringify(errorResponse)}\n\n`);
                    req.res.end();
                }
            })
        );
    } finally {
        // Process next batch immediately if there are more requests
        if (streamRequestQueue.length > 0) {
            processStreamBatch();
        }
    }
}

// Update the OpenAI-compatible endpoint to increment activeRequests
app.post('/v1/chat/completions', async (req, res) => {
    try {
        // Increment activeRequests to track pending requests more accurately
        activeRequests++;
        
        // Check various conditions for OpenRouter fallback:
        // 1. Queue size exceeds limit, or
        // 2. Estimated tokens exceed context length, or
        // 3. VLLM server is unavailable, or
        // 4. VLLM waiting requests exceed MAX_VLLM_WAITING_REQUESTS
        const totalQueueSize = requestQueue.length + streamRequestQueue.length;
        const estimatedTokens = estimateTokens(req.body.messages) + (req.body.max_tokens || DEFAULT_MAX_TOKENS);
        const tooManyWaitingRequests = vllmWaitingRequests >= MAX_VLLM_WAITING_REQUESTS;
        
        // Forward to OpenRouter if any fallback condition is met
        if (totalQueueSize >= MAX_QUEUE_SIZE || 
            estimatedTokens > MAX_CONTEXT_LENGTH || 
            !isVLLMAvailable || 
            tooManyWaitingRequests) {
            
            const reason = !isVLLMAvailable ? 'VLLM unavailable' : 
                           (totalQueueSize >= MAX_QUEUE_SIZE ? 'Queue full' : 
                           (estimatedTokens > MAX_CONTEXT_LENGTH ? 'Context length exceeded' : 
                           'Too many waiting requests'));
            
            console.log(`Forwarding to OpenRouter: ${reason}`);
            
            if (req.body.stream) {
                try {
                    const openRouterStream = await forwardToOpenRouter(req, true);
                    
                    res.setHeader('Content-Type', 'text/event-stream');
                    res.setHeader('Cache-Control', 'no-cache');
                    res.setHeader('Connection', 'keep-alive');
                    
                    let buffer = '';
                    let isEnded = false;
                    
                    openRouterStream.on('data', (chunk) => {
                        if (isEnded) return;
                        
                        buffer += chunk.toString();
                        const lines = buffer.split('\n');
                        buffer = lines.pop() || '';
                        
                        lines.forEach(line => {
                            if (line.trim() === '') return;
                            if (line.startsWith('data: ')) {
                                const data = line.slice(6);
                                if (data === '[DONE]') {
                                    if (!isEnded) {
                                        res.write('data: [DONE]\n\n');
                                        res.end();
                                        isEnded = true;
                                    }
                                } else {
                                    try {
                                        // Normalize the OpenRouter response to match standard OpenAI format
                                        const parsed = JSON.parse(data);
                                        const normalized = normalizeStreamChunk(parsed, 'openrouter');
                                        if (!isEnded) {
                                            res.write(`data: ${JSON.stringify(normalized)}\n\n`);
                                        }
                                    } catch (e) {
                                        console.error('Error parsing OpenRouter stream data:', e);
                                    }
                                }
                            }
                        });
                    });
                    
                    openRouterStream.on('end', () => {
                        if (!isEnded) {
                            res.write('data: [DONE]\n\n');
                            res.end();
                            isEnded = true;
                        }
                    });
                    
                    openRouterStream.on('error', (error) => {
                        if (!isEnded) {
                            res.write(`data: ${JSON.stringify({ error: error.message })}\n\n`);
                            res.end();
                            isEnded = true;
                        }
                    });
                    
                    res.on('close', () => {
                        isEnded = true;
                    });
                } catch (error) {
                    // Standardize error format
                    const errorResponse = {
                        error: {
                            message: error.message || 'OpenRouter error',
                            type: 'openrouter_error'
                        }
                    };
                    res.write(`data: ${JSON.stringify(errorResponse)}\n\n`);
                    res.end();
                }
            } else {
                try {
                    const openRouterResponse = await forwardToOpenRouter(req);
                    res.json(openRouterResponse);
                } catch (error) {
                    // Standardize error format
                    res.status(500).json({
                        error: {
                            message: error.response?.data || error.message,
                            type: 'openrouter_error'
                        }
                    });
                }
            }
            return;
        }

        if (req.body.stream) {
            streamRequestQueue.push({ body: req.body, res });
            if (streamRequestQueue.length === 1) {
                processStreamBatch();
            }
        } else {
            const response = await new Promise((resolve, reject) => {
                requestQueue.push({
                    body: req.body,
                    resolve,
                    reject
                });
                
                if (requestQueue.length === 1) {
                    processBatch();
                }
            });
            
            res.json(response);
        }
    } catch (error) {
        if (!req.body.stream) {
            // Standardize error format
            res.status(500).json({
                error: {
                    message: error.response?.data || error.message,
                    type: 'internal_server_error'
                }
            });
        }
    } finally {
        // Ensure activeRequests is decremented when the request is fully processed
        activeRequests--;
    }
});

// Start batch processing intervals - use separate intervals with faster timing for processing
setInterval(processBatch, BATCH_INTERVAL);
setInterval(processStreamBatch, BATCH_INTERVAL);

// Add a route to get current server status
app.get('/status', (req, res) => {
    res.json({
        status: 'ok',
        metrics: {
            activeRequests,
            queueSize: {
                total: requestQueue.length + streamRequestQueue.length,
                nonStream: requestQueue.length,
                stream: streamRequestQueue.length
            },
            vllm: {
                available: isVLLMAvailable,
                waitingRequests: vllmWaitingRequests
            }
        }
    });
});

// Start the server
app.listen(port, () => {
    console.log(`Batcher server running at http://localhost:${port}`);
});
