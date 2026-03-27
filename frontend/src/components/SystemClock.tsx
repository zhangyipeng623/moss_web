import { twMerge } from 'tailwind-merge';

interface SystemClockProps {
    time: string;
    mode: string;
    className?: string;
}

export function SystemClock({ time, mode, className }: SystemClockProps) {
    const displayTime = time ? time.split('.')[0] : '';

    return (
        <div className={twMerge("fixed top-4 left-4 z-50 bg-white/80 backdrop-blur-md px-4 py-2 rounded-lg border border-slate-200 shadow-md font-display text-sm text-cyan-700", className)}>
            <div className="flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
                <span className="font-medium tracking-wide">{displayTime || 'Loading...'}</span>
                {mode && <span className="text-xs text-slate-400 font-normal border border-slate-200 px-1.5 py-0.5 rounded bg-slate-50 uppercase tracking-wider">{mode}</span>}
            </div>
        </div>
    );
}
