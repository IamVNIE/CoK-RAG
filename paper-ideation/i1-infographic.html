<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CoKRAG: Agentic Chain-of-Knowledge Graph RAG</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;900&display=swap" rel="stylesheet">
    <style>
        body {
            font-family: 'Inter', sans-serif;
        }
        .chart-container {
            position: relative;
            width: 100%;
            max-width: 600px;
            margin-left: auto;
            margin-right: auto;
            height: 320px;
            max-height: 400px;
        }
        @media (min-width: 768px) {
            .chart-container {
                height: 350px;
            }
        }
        .flow-line {
            position: relative;
            height: 2px;
            background-color: #FF6B6B;
            width: 100%;
        }
        .flow-line::after {
            content: '▼';
            position: absolute;
            bottom: -11px;
            left: 50%;
            transform: translateX(-50%);
            color: #FF6B6B;
            font-size: 1.2rem;
        }
        .flow-line-horiz {
            position: relative;
            width: 2px;
            background-color: #FF6B6B;
            height: 100%;
        }
        .flow-line-horiz::after {
            content: '▶';
            position: absolute;
            top: 50%;
            right: -9px;
            transform: translateY(-50%);
            color: #FF6B6B;
            font-size: 1.2rem;
        }
        .gradient-bg {
            background: linear-gradient(135deg, #4D96FF, #6BCB77);
        }
        .card-glow {
            box-shadow: 0 0 15px rgba(77, 150, 255, 0.5), 0 0 30px rgba(107, 203, 119, 0.3);
        }
    </style>
</head>
<body class="bg-gray-900 text-white">

    <div class="container mx-auto p-4 md:p-8">
        
        <header class="text-center mb-16">
            <h1 class="text-4xl md:text-6xl font-black uppercase tracking-wider bg-clip-text text-transparent gradient-bg">CoKRAG</h1>
            <p class="text-xl md:text-2xl font-light text-gray-300 mt-2">Agentic Chain-of-Knowledge Graph RAG</p>
            <p class="max-w-3xl mx-auto text-gray-400 mt-4">A new frontier in Retrieval-Augmented Generation, combining autonomous agents, deep reasoning, and structured knowledge to answer complex questions with unparalleled accuracy and transparency.</p>
        </header>

        <section id="problem" class="mb-20">
            <h2 class="text-3xl font-bold text-center mb-4">The Challenge with Standard RAG</h2>
            <p class="text-center max-w-2xl mx-auto text-gray-400 mb-12">Traditional RAG models struggle with complex, multi-step questions. They often retrieve irrelevant document chunks and lack the deep reasoning needed to synthesize a coherent answer, leading to factual errors and "hallucinations".</p>
            <div class="grid grid-cols-1 md:grid-cols-3 gap-8 text-center">
                <div class="bg-gray-800 p-6 rounded-lg border border-red-500/50">
                    <p class="text-5xl mb-4">⚠️</p>
                    <h3 class="text-xl font-semibold text-red-400 mb-2">High Hallucination Risk</h3>
                    <p class="text-gray-400">Generates plausible but incorrect information when faced with queries requiring deep domain knowledge.</p>
                </div>
                <div class="bg-gray-800 p-6 rounded-lg border border-yellow-500/50">
                    <p class="text-5xl mb-4">🧩</p>
                    <h3 class="text-xl font-semibold text-yellow-400 mb-2">Poor Multi-Hop Reasoning</h3>
                    <p class="text-gray-400">Fails to connect disparate pieces of information across multiple documents or steps.</p>
                </div>
                <div class="bg-gray-800 p-6 rounded-lg border-blue-500/50">
                    <p class="text-5xl mb-4">🔍</p>
                    <h3 class="text-xl font-semibold text-blue-400 mb-2">Lack of Explainability</h3>
                    <p class="text-gray-400">Provides answers without a clear, traceable reasoning path, making it a "black box".</p>
                </div>
            </div>
        </section>

        <section id="architecture" class="mb-20">
            <h2 class="text-3xl font-bold text-center mb-12">The CoKRAG Architecture: A Symphony of Components</h2>
             <div class="bg-gray-800 p-6 md:p-10 rounded-2xl card-glow">
                <div class="flex flex-col items-center">
                    
                    <div class="bg-gray-700 p-4 rounded-lg text-center shadow-lg w-full md:w-3/4">
                        <p class="text-lg font-semibold">User Query</p>
                        <p class="text-sm text-gray-400">"Which drug, developed by a company that acquired a firm specializing in gene editing, has the highest efficacy for treating Type II Diabetes?"</p>
                    </div>

                    <div class="w-full flex justify-center my-4"><div class="flow-line"></div></div>

                    <div class="bg-indigo-500/20 border border-indigo-400 text-indigo-300 p-4 rounded-lg text-center shadow-lg w-full md:w-3/4">
                        <p class="text-lg font-bold">1. Agentic Planner</p>
                        <p class="text-sm">Decomposes the query into a multi-step execution plan.</p>
                    </div>

                    <div class="w-full flex justify-center my-4"><div class="flow-line"></div></div>

                    <div class="w-full p-4 rounded-lg bg-gray-900/50 text-center">
                        <p class="text-lg font-bold text-yellow-300 mb-4">2. Chain-of-Knowledge Reasoning Loop</p>
                        <div class="flex flex-col md:flex-row items-stretch justify-center gap-4">
                            <div class="flex-1 bg-gray-700 p-4 rounded-lg text-center">
                                <h4 class="font-semibold">Step A: Find companies that acquired gene editing firms</h4>
                                <p class="text-xs text-gray-400 mt-2">TOOL: Knowledge Graph Traversal</p>
                            </div>
                            <div class="hidden md:flex items-center justify-center"><div class="flow-line-horiz"></div></div>
                             <div class="md:hidden w-full flex justify-center my-4"><div class="flow-line"></div></div>
                            <div class="flex-1 bg-gray-700 p-4 rounded-lg text-center">
                                <h4 class="font-semibold">Step B: Identify drugs for Type II Diabetes by those companies</h4>
                                <p class="text-xs text-gray-400 mt-2">TOOL: Semantic Search + KG Filter</p>
                            </div>
                            <div class="hidden md:flex items-center justify-center"><div class="flow-line-horiz"></div></div>
                             <div class="md:hidden w-full flex justify-center my-4"><div class="flow-line"></div></div>
                            <div class="flex-1 bg-gray-700 p-4 rounded-lg text-center">
                                <h4 class="font-semibold">Step C: Compare efficacy data for final candidates</h4>
                                <p class="text-xs text-gray-400 mt-2">TOOL: Data Retrieval from KG Nodes</p>
                            </div>
                        </div>
                    </div>
                    
                    <div class="w-full flex justify-center my-4"><div class="flow-line"></div></div>

                    <div class="bg-green-500/20 border border-green-400 text-green-300 p-4 rounded-lg text-center shadow-lg w-full md:w-3/4">
                        <p class="text-lg font-bold">3. Synthesized, Traceable Answer</p>
                        <p class="text-sm">Generates a final response grounded in the retrieved evidence chain.</p>
                    </div>

                </div>
            </div>
        </section>

        <section id="results" class="mb-20">
            <h2 class="text-3xl font-bold text-center mb-4">Performance Benchmarks</h2>
            <p class="text-center max-w-2xl mx-auto text-gray-400 mb-12">CoKRAG was evaluated against leading RAG models on the MINTQA multi-hop question-answering benchmark. The results demonstrate significant improvements across key metrics.</p>

            <div class="grid grid-cols-1 md:grid-cols-2 gap-8 md:gap-12">
                <div class="bg-gray-800 p-6 rounded-lg card-glow">
                    <h3 class="text-xl font-semibold text-center mb-4 text-green-300">Answer Faithfulness</h3>
                    <p class="text-sm text-center text-gray-400 mb-6">Measures if the answer is factually grounded in the provided context, avoiding hallucinations. Higher is better.</p>
                    <div class="chart-container">
                        <canvas id="faithfulnessChart"></canvas>
                    </div>
                </div>
                <div class="bg-gray-800 p-6 rounded-lg card-glow">
                    <h3 class="text-xl font-semibold text-center mb-4 text-blue-300">Context Precision</h3>
                    <p class="text-sm text-center text-gray-400 mb-6">Measures if the retrieved information is relevant to the user's query. Higher is better.</p>
                    <div class="chart-container">
                        <canvas id="precisionChart"></canvas>
                    </div>
                </div>
            </div>
        </section>
        
        <section id="comparison" class="mb-20">
            <h2 class="text-3xl font-bold text-center mb-4">Qualitative Trait Comparison</h2>
            <p class="text-center max-w-2xl mx-auto text-gray-400 mb-12">CoKRAG excels not just in quantitative metrics but also in qualitative capabilities crucial for enterprise-grade AI applications.</p>
            <div class="bg-gray-800 p-6 rounded-lg card-glow">
                <div class="chart-container h-[400px] md:h-[450px]">
                     <canvas id="radarChart"></canvas>
                </div>
            </div>
        </section>

        <section id="future" class="text-center">
            <h2 class="text-3xl font-bold mb-4">The Future is Agentic & Connected</h2>
            <p class="max-w-3xl mx-auto text-gray-400 mb-8">CoKRAG represents a paradigm shift towards more intelligent, reliable, and transparent AI. Future work will focus on dynamic knowledge graph updates, more sophisticated agentic planning, and scaling the reasoning process for even more complex, real-world scenarios.</p>
            <div class="flex justify-center gap-4">
                 <div class="bg-yellow-500/20 border border-yellow-400 text-yellow-300 py-3 px-6 rounded-lg">
                    <h4 class="font-bold">Dynamic KG Updates</h4>
                 </div>
                 <div class="bg-purple-500/20 border border-purple-400 text-purple-300 py-3 px-6 rounded-lg">
                    <h4 class="font-bold">Advanced Planning</h4>
                 </div>
                 <div class="bg-pink-500/20 border border-pink-400 text-pink-300 py-3 px-6 rounded-lg">
                    <h4 class="font-bold">Web-Scale Reasoning</h4>
                 </div>
            </div>
        </section>

    </div>

    <script>
        const vibrantPalette = {
            blue: '#4D96FF',
            green: '#6BCB77',
            yellow: '#FFD93D',
            red: '#FF6B6B',
            purple: '#9B5DE5',
            gray: 'rgba(209, 213, 219, 0.5)'
        };

        function wrapLabels(label) {
            const maxLen = 16;
            if (typeof label !== 'string' || label.length <= maxLen) {
                return label;
            }
            const words = label.split(' ');
            const lines = [];
            let currentLine = '';
            words.forEach(word => {
                if ((currentLine + ' ' + word).trim().length > maxLen) {
                    lines.push(currentLine.trim());
                    currentLine = word;
                } else {
                    currentLine = (currentLine + ' ' + word).trim();
                }
            });
            if (currentLine) {
                lines.push(currentLine.trim());
            }
            return lines;
        }

        const tooltipTitleCallback = (tooltipItems) => {
            const item = tooltipItems[0];
            let label = item.chart.data.labels[item.dataIndex];
            if (Array.isArray(label)) {
                return label.join(' ');
            } else {
                return label;
            }
        };

        const chartDefaultOptions = {
            maintainAspectRatio: false,
            responsive: true,
            plugins: {
                legend: {
                    labels: {
                        color: '#D1D5DB',
                        font: {
                            size: 14
                        }
                    }
                },
                tooltip: {
                    callbacks: {
                        title: tooltipTitleCallback
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    },
                    ticks: {
                        color: '#D1D5DB',
                        font: {
                           size: 12
                        }
                    }
                },
                x: {
                    grid: {
                        display: false
                    },
                    ticks: {
                        color: '#D1D5DB',
                         font: {
                           size: 12
                        }
                    }
                }
            }
        };
        
        const faithfulnessCtx = document.getElementById('faithfulnessChart').getContext('2d');
        new Chart(faithfulnessCtx, {
            type: 'bar',
            data: {
                labels: ['Standard RAG', 'GraphRAG', 'PathRAG', 'LightRAG', 'CoKRAG'].map(wrapLabels),
                datasets: [{
                    label: 'Faithfulness Score',
                    data: [0.68, 0.79, 0.82, 0.81, 0.94],
                    backgroundColor: vibrantPalette.green,
                    borderColor: '#fff',
                    borderWidth: 2,
                    borderRadius: 5
                }]
            },
            options: chartDefaultOptions
        });

        const precisionCtx = document.getElementById('precisionChart').getContext('2d');
        new Chart(precisionCtx, {
            type: 'bar',
            data: {
                labels: ['Standard RAG', 'GraphRAG', 'PathRAG', 'LightRAG', 'CoKRAG'].map(wrapLabels),
                datasets: [{
                    label: 'Context Precision Score',
                    data: [0.72, 0.85, 0.84, 0.88, 0.96],
                    backgroundColor: vibrantPalette.blue,
                    borderColor: '#fff',
                    borderWidth: 2,
                    borderRadius: 5
                }]
            },
            options: chartDefaultOptions
        });

        const radarCtx = document.getElementById('radarChart').getContext('2d');
        new Chart(radarCtx, {
            type: 'radar',
            data: {
                labels: ['Reasoning Depth', 'Explainability', 'Adaptability', 'Factual Accuracy', ['Handling', 'Ambiguity'], 'Automation'],
                datasets: [{
                    label: 'CoKRAG',
                    data: [9.5, 9, 8.5, 9.4, 8, 9],
                    backgroundColor: 'rgba(77, 150, 255, 0.2)',
                    borderColor: vibrantPalette.blue,
                    pointBackgroundColor: vibrantPalette.blue,
                    pointBorderColor: '#fff',
                    pointHoverBackgroundColor: '#fff',
                    pointHoverBorderColor: vibrantPalette.blue
                }, {
                    label: 'Average Graph RAG',
                    data: [7, 6, 6.5, 7.9, 6, 5],
                    backgroundColor: 'rgba(107, 203, 119, 0.2)',
                    borderColor: vibrantPalette.green,
                    pointBackgroundColor: vibrantPalette.green,
                    pointBorderColor: '#fff',
                    pointHoverBackgroundColor: '#fff',
                    pointHoverBorderColor: vibrantPalette.green
                }]
            },
            options: {
                maintainAspectRatio: false,
                responsive: true,
                plugins: {
                    legend: {
                         labels: {
                            color: '#D1D5DB',
                             font: {
                                size: 14
                            }
                        }
                    },
                    tooltip: {
                         callbacks: {
                            title: tooltipTitleCallback
                        }
                    }
                },
                scales: {
                    r: {
                        angleLines: {
                            color: 'rgba(255, 255, 255, 0.2)'
                        },
                        grid: {
                            color: 'rgba(255, 255, 255, 0.2)'
                        },
                        pointLabels: {
                            color: '#F9FAFB',
                             font: {
                                size: 12
                            }
                        },
                        ticks: {
                            color: '#D1D5DB',
                            backdropColor: 'rgba(0,0,0,0.5)',
                            stepSize: 2,
                        },
                        suggestedMin: 0,
                        suggestedMax: 10
                    }
                }
            }
        });
    </script>
</body>
</html>
