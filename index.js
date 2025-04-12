const express = require('express');
const bodyParser = require('body-parser');
const axios = require('axios');
const dotenv = require('dotenv');

dotenv.config();

const app = express();
const port = process.env.PORT || 3000;

// Configuration
const VLLM_ENDPOINT = `${process.env.VLLM_ENDPOINT || 'http://localhost:8080'}/v1/chat/completions`;
const MODEL = process.env.MODEL || 'nvidia/Llama-3.1-Nemotron-Nano-8B-v1';
const MAX_CONTEXT_LENGTH = process.env.MAX_CONTEXT_LENGTH || 8192; // Maximum context length for the model
const DEFAULT_MAX_TOKENS = 512; // Default max tokens for completion
const BATCH_SIZE = 50; // Number of requests to process in each batch

// Request queues
let requestQueue = [];
let streamRequestQueue = [];
let isProcessing = false;
let isStreamProcessing = false;

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

// Function to process non-streaming batch
async function processBatch() {
    if (requestQueue.length === 0 || isProcessing) return;

    isProcessing = true;
    const batch = requestQueue.splice(0, BATCH_SIZE);

    try {
        // Process all requests in the batch in parallel
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

        // Send all requests in parallel
        const responses = await Promise.all(
            requests.map(async (r, index) => {
                try {
                    const response = await axios.post(VLLM_ENDPOINT, r);
                    return { index, data: response.data };
                } catch (error) {
                    return { index, error: error.response?.data || error.message };
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
        // Handle any errors in parallel
        await Promise.all(
            batch.map(req => req.reject(error.response?.data || error.message))
        );
    } finally {
        isProcessing = false;
        // Process next batch if there are remaining requests
        if (requestQueue.length > 0) {
            processBatch();
        }
    }
}

// Function to process streaming batch
async function processStreamBatch() {
    if (streamRequestQueue.length === 0 || isStreamProcessing) return;

    isStreamProcessing = true;
    const batch = streamRequestQueue.splice(0, BATCH_SIZE);

    try {
        // Process all requests in the batch in parallel
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

        // Send all requests in parallel
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
                }
            })
        );

        // Handle all responses in parallel
        await Promise.all(
            responses.map(({ index, data, error }) => {
                const { res } = batch[index];
                let isEnded = false;

                // Set SSE headers
                res.setHeader('Content-Type', 'text/event-stream');
                res.setHeader('Cache-Control', 'no-cache');
                res.setHeader('Connection', 'keep-alive');

                if (error) {
                    if (!isEnded) {
                        const errorMessage = typeof error === 'string' ? error : JSON.stringify(error);
                        res.write(`data: ${JSON.stringify({ error: errorMessage })}\n\n`);
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
                                        if (!isEnded) {
                                            res.write(`data: ${JSON.stringify(parsed)}\n\n`);
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
                                if (!isEnded) {
                                    res.write(`data: ${JSON.stringify(parsed)}\n\n`);
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
                            res.write(`data: ${JSON.stringify({ error: error.message })}\n\n`);
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
        // Handle any errors in parallel
        await Promise.all(
            batch.map(req => {
                if (!req.res.writableEnded) {
                    const errorMessage = error.response?.data || error.message;
                    req.res.write(`data: ${JSON.stringify({ error: errorMessage })}\n\n`);
                    req.res.end();
                }
            })
        );
    } finally {
        isStreamProcessing = false;
        // Process next batch if there are remaining requests
        if (streamRequestQueue.length > 0) {
            processStreamBatch();
        }
    }
}

// OpenAI-compatible endpoint
app.post('/v1/chat/completions', async (req, res) => {
    try {
        if (req.body.stream) {
            // Add to stream queue and process if not already processing
            streamRequestQueue.push({ body: req.body, res });
            if (!isStreamProcessing) {
                processStreamBatch();
            }
        } else {
            // Non-streaming request - wait for the response before sending
            const response = await new Promise((resolve, reject) => {
                requestQueue.push({
                    body: req.body,
                    resolve,
                    reject
                });
                // Process if not already processing
                if (!isProcessing) {
                    processBatch();
                }
            });
            
            // Only send response after getting the VLLM response
            res.json(response);
        }
    } catch (error) {
        if (!req.body.stream) {
            res.status(500).json({
                error: {
                    message: error.response?.data || error.message,
                    type: 'internal_server_error'
                }
            });
        }
    }
});

// Start the server
app.listen(port, () => {
    console.log(`Batcher server running at http://localhost:${port}`);
});
