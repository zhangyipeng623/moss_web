import { useState, useEffect, useRef } from 'react';
import type { Trace } from '../types';
import { ChevronRight, ChevronLeft, Search, Info, MessageSquare, Heart, Repeat, PenTool, MousePointer2 } from 'lucide-react';
import { twMerge } from 'tailwind-merge';

interface TracePanelProps {
    className?: string;
    traces: Trace[];
}

export function TracePanel({
    className,
    traces,
}: TracePanelProps) {
    const [isOpen, setIsOpen] = useState(true);
    const [width, setWidth] = useState(320);
    const [filter, setFilter] = useState('');
    const [isResizing, setIsResizing] = useState(false);
    const sidebarRef = useRef<HTMLDivElement>(null);

    // Resizing logic
    const startResizing = (e: React.MouseEvent) => {
        setIsResizing(true);
        e.preventDefault();
    };

    useEffect(() => {
        const handleMouseMove = (e: MouseEvent) => {
            if (!isResizing) return;
            const newWidth = window.innerWidth - e.clientX;
            if (newWidth > 200 && newWidth < 600) {
                setWidth(newWidth);
            }
        };

        const handleMouseUp = () => {
            setIsResizing(false);
        };

        if (isResizing) {
            window.addEventListener('mousemove', handleMouseMove);
            window.addEventListener('mouseup', handleMouseUp);
        }

        return () => {
            window.removeEventListener('mousemove', handleMouseMove);
            window.removeEventListener('mouseup', handleMouseUp);
        };
    }, [isResizing]);

    const filteredTraces = traces.filter(t =>
        (t.action_type && t.action_type.toLowerCase().includes(filter.toLowerCase())) ||
        (t.user_nickname && t.user_nickname.toLowerCase().includes(filter.toLowerCase()))
    );

    const renderTraceIcon = (type: string) => {
        switch (type.toLowerCase()) {
            case 'create_post': return <PenTool size={12} className="text-cyan-600" />;
            case 'create_comment': return <MessageSquare size={12} className="text-blue-500" />;
            case 'like_post':
            case 'like_comment': return <Heart size={12} className="text-rose-500" />;
            case 'repost':
            case 'quote': return <Repeat size={12} className="text-emerald-500" />;
            case 'do_nothing': return <MousePointer2 size={12} className="text-slate-400" />;
            default: return <Info size={12} className="text-slate-500" />;
        }
    };

    const renderTraceContent = (trace: Trace) => {
        const details = trace.action_details || {};
        const type = trace.action_type.toLowerCase();

        switch (type) {
            case 'create_post':
                return (
                    <span>
                        published a post: <span className="italic text-slate-600">"{String(details.content || '')}"</span>
                    </span>
                );
            case 'create_comment':
                return (
                    <span>
                        commented on post #{String(details.post_id)}: <span className="italic text-slate-600">"{String(details.content || '')}"</span>
                    </span>
                );
            case 'like_post':
                return <span>liked post #{String(details.post_id)}</span>;
            case 'like_comment':
                return <span>liked comment #{String(details.comment_id)}</span>;
            case 'repost':
                return (
                    <span>
                        reposted post #{String(details.original_post_id)}
                        {!!details.content && <span> with: <span className="italic text-slate-600">"{String(details.content)}"</span></span>}
                    </span>
                );
            case 'quote':
                return (
                    <span>
                        quoted post #{String(details.original_post_id)} with: <span className="italic text-slate-600">"{String(details.content)}"</span>
                    </span>
                );
            case 'do_nothing':
                return <span className="text-slate-400">is idle</span>;
            default:
                return <span className="font-mono text-[10px] text-slate-500">{JSON.stringify(details)}</span>;
        }
    };

    if (!isOpen) {
        return (
            <div className="fixed right-4 top-24 bottom-24 w-8 bg-white/90 backdrop-blur-md border border-slate-200 rounded-l-lg flex flex-col items-center py-4 z-40 transition-all duration-300 shadow-xl">
                <button onClick={() => setIsOpen(true)} className="p-1 hover:bg-slate-100 rounded mb-4 text-slate-500">
                    <ChevronLeft size={16} />
                </button>
                <div className="writing-vertical text-xs text-slate-400 tracking-wider font-bold" style={{ writingMode: 'vertical-rl' }}>SYSTEM TRACES</div>
            </div>
        );
    }

    return (
        <div
            ref={sidebarRef}
            style={{ width: `${width}px` }}
            className={twMerge("fixed right-4 top-24 bottom-24 bg-white/95 backdrop-blur-md border border-slate-200 rounded-xl flex flex-col z-40 transition-width duration-0 hidden md:flex shadow-2xl", className)}
        >
            {/* Drag Handle */}
            <div
                className="absolute left-0 top-0 w-4 h-full cursor-ew-resize hover:bg-cyan-500/10 z-50 flex items-center justify-center group"
                onMouseDown={startResizing}
            >
                <div className="w-1 h-8 bg-slate-200 rounded-full group-hover:bg-cyan-400/50 transition-colors" />
            </div>

            {/* Header */}
            <div className="flex items-center justify-between p-3 border-b border-slate-100 bg-slate-50/50 rounded-t-xl pl-6">
                <h3 className="text-sm font-bold font-display text-slate-700">System Traces</h3>
                <div className="flex items-center gap-1">
                    <button onClick={() => setIsOpen(false)} className="p-1 hover:bg-slate-200 rounded text-slate-500">
                        <ChevronRight size={16} />
                    </button>
                </div>
            </div>

            {/* Toolbar */}
            <div className="p-2 border-b border-slate-100 bg-white/50">
                <div className="relative">
                    <Search size={12} className="absolute left-2 top-1/2 -translate-y-1/2 text-slate-400" />
                    <input
                        type="text"
                        placeholder="筛选轨迹..."
                        value={filter}
                        onChange={(e) => setFilter(e.target.value)}
                        className="w-full bg-slate-100 border border-slate-200 rounded px-2 py-1 pl-7 text-xs text-slate-600 focus:outline-none focus:border-cyan-500/50 focus:bg-white transition-colors"
                    />
                </div>
            </div>

            {/* Log List */}
            <div className="flex-1 overflow-y-auto p-2 space-y-2 font-mono text-xs">
                {filteredTraces.map((trace, i) => (
                    <div key={trace.id || i} className="p-2 rounded bg-slate-50 border border-slate-100 hover:bg-white hover:shadow-sm transition-all">
                        <div className="flex items-start gap-2">
                            <span className="mt-0.5">
                                {renderTraceIcon(trace.action_type)}
                            </span>
                            <div className="flex-1 min-w-0">
                                <div className="flex justify-between items-center mb-1">
                                    <span className="font-bold uppercase text-[10px] text-slate-500">
                                        {trace.action_type.replace('_', ' ')}
                                    </span>
                                    <span className="text-[10px] text-slate-400">{new Date(trace.created_at).toLocaleTimeString()}</span>
                                </div>
                                <p className="text-slate-600 break-words leading-relaxed line-clamp-3 overflow-hidden">
                                    <span className="font-bold text-slate-800">{trace.user_nickname}</span> {renderTraceContent(trace)}
                                </p>
                            </div>
                        </div>
                    </div>
                ))}
                {filteredTraces.length === 0 && (
                    <div className="text-center text-slate-400 py-8">暂无轨迹</div>
                )}
            </div>
        </div>
    );
}
