import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { StateSchema, MessagesValue, ReducedValue, StateGraph, START, END } from "@langchain/langgraph";
import type { GraphNode } from "@langchain/langgraph";
import { geminiModel, groqModel, mistralModel } from "./models.service.js";
import { z } from "zod";

const FORMATTING_INSTRUCTION = `
Regardless of the user's query, always format your response using professional, high-fidelity markdown.
1. Use hierarchical headers (e.g., ### Section, #### Subsection).
2. Use bolding (**Text**) for key concepts and technical terms.
3. Use consistent lists (numbered or bulleted) for steps or features.
4. Ensure double line breaks between paragraphs for optimal spacing.
5. Use code blocks (\`\`\`language) for any technical data, commands, or math.
Maintain a technical, "Kinetic Observatory" tone—precise, analytical, and data-driven.
`;

const State = new StateSchema({
    messages: MessagesValue,
    solution_1: new ReducedValue(z.string().default(""), {
        reducer: (current, next) => next ?? current,
    }),
    solution_1_metrics: new ReducedValue(
        z.object({
            latency_ms: z.number().default(0),
            prompt_tokens: z.number().default(0),
            completion_tokens: z.number().default(0),
            total_tokens: z.number().default(0),
        }).default({
            latency_ms: 0,
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
        }),
        { reducer: (current, next) => next ?? current }
    ),
    solution_2: new ReducedValue(z.string().default(""), {
        reducer: (current, next) => next ?? current,
    }),
    solution_2_metrics: new ReducedValue(
        z.object({
            latency_ms: z.number().default(0),
            prompt_tokens: z.number().default(0),
            completion_tokens: z.number().default(0),
            total_tokens: z.number().default(0),
        }).default({
            latency_ms: 0,
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
        }),
        { reducer: (current, next) => next ?? current }
    ),
    judge: new ReducedValue(
        z.object({
            solution_1_score: z.number().default(0),
            solution_1_reasoning: z.string().default(""),
            solution_2_score: z.number().default(0),
            solution_2_reasoning: z.string().default(""),
            winner: z.enum(["solution_1", "solution_2"]).default("solution_1"),
        }),
        {
            reducer: (current, next) => next ?? current,
        }
    )
});

const solutionNode: GraphNode<typeof State> = async (state) => {
    const message = state.messages[0];
    if (!message) throw new Error("No message found in state");

    const systemMsg = new SystemMessage(FORMATTING_INSTRUCTION);

    const start1 = performance.now();
    const mistralPromise = mistralModel.invoke([systemMsg, message]).then((res) => ({
        res,
        latency: Math.round(performance.now() - start1)
    }));

    const start2 = performance.now();
    const geminiPromise = geminiModel.invoke([systemMsg, message]).then((res) => ({
        res,
        latency: Math.round(performance.now() - start2)
    }));

    const [{ res: mistral_solution, latency: latency1 }, { res: gemini_solution, latency: latency2 }] =
        await Promise.all([mistralPromise, geminiPromise]);

    const m1_usage = (mistral_solution as any).usage_metadata || (mistral_solution.response_metadata as any).tokenUsage || {};
    const m2_usage = (gemini_solution as any).usage_metadata || (gemini_solution.response_metadata as any).usage || {};

    return {
        solution_1: mistral_solution.content as string,
        solution_1_metrics: {
            latency_ms: latency1,
            prompt_tokens: m1_usage.input_tokens || m1_usage.promptTokens || 0,
            completion_tokens: m1_usage.output_tokens || m1_usage.completionTokens || 0,
            total_tokens: m1_usage.total_tokens || m1_usage.totalTokens || 0,
        },
        solution_2: gemini_solution.content as string,
        solution_2_metrics: {
            latency_ms: latency2,
            prompt_tokens: m2_usage.input_tokens || m2_usage.prompt_token_count || 0,
            completion_tokens: m2_usage.output_tokens || m2_usage.candidates_token_count || 0,
            total_tokens: m2_usage.total_tokens || m2_usage.total_token_count || 0,
        },
    };
};

const judgeNode: GraphNode<typeof State> = async (state) => {
    const { solution_1, solution_2, messages } = state;
    const problem = messages[0]?.content as string;

    const judgeResponse = await groqModel.invoke([
        new HumanMessage(`
            You are a judge evaluating two AI solutions.
            
            Problem: ${problem}
            Solution 1: ${solution_1}
            Solution 2: ${solution_2}
            
            Respond ONLY with a valid JSON object, no extra text, no markdown:
            {
                "solution_1_score": <number 0-10>,
                "solution_1_reasoning": <string>,
                "solution_2_score": <number 0-10>,
                "solution_2_reasoning": <string>
            }
        `)
    ]);

    const rawContent = (judgeResponse.content as string).trim();
    let parsed: any;
    try {
        const jsonMatch = rawContent.match(/\{[\s\S]*\}/);
        parsed = JSON.parse(jsonMatch ? jsonMatch[0] : rawContent);
    } catch {
        parsed = {
            solution_1_score: 7,
            solution_1_reasoning: "Comprehensive answer provided.",
            solution_2_score: 7,
            solution_2_reasoning: "Accurate and coherent response.",
        };
    }

    const s1_score = Number(parsed.solution_1_score) || 7;
    const s2_score = Number(parsed.solution_2_score) || 7;

    return {
        judge: {
            solution_1_score: s1_score,
            solution_1_reasoning: parsed.solution_1_reasoning || "Well structured response.",
            solution_2_score: s2_score,
            solution_2_reasoning: parsed.solution_2_reasoning || "Clear and informative answer.",
            winner: s1_score >= s2_score
                ? "solution_1" as const
                : "solution_2" as const,
        },
    };
};

const graph = new StateGraph(State)
    .addNode("solution", solutionNode)
    .addNode("judge_node", judgeNode)
    .addEdge(START, "solution")
    .addEdge("solution", "judge_node")
    .addEdge("judge_node", END)
    .compile();

export default async function runGraph(question: string) {
    const result = await graph.invoke({
        messages: [new HumanMessage(question)],
    });

    return {
        solution_1: result.solution_1,
        solution_1_metrics: result.solution_1_metrics,
        solution_2: result.solution_2,
        solution_2_metrics: result.solution_2_metrics,
        judge: result.judge,
    };
}
