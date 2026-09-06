import { useState, useEffect, useRef } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { clsx } from 'clsx';
import { SystemClock } from './components/SystemClock';
import { TracePanel } from './components/TracePanel';
import { PostList } from './components/PostList';
import { PostDetail } from './pages/PostDetail';
import { getPosts, getTraces, getTime } from './lib/api';
import type { Post, Trace } from './types';

type SortKey = 'new' | 'hot' | 'reply' | 'like' | 'retweet' | 'quote';

const SORT_OPTIONS: { value: SortKey; label: string }[] = [
  { value: 'new', label: '最新' },
  { value: 'hot', label: '最热' },
  { value: 'reply', label: '评论最多' },
  { value: 'like', label: '点赞最多' },
  { value: 'retweet', label: '转发最多' },
  { value: 'quote', label: '引用最多' },
];

function Home({ posts, systemTime, onLoadMore, hasMore, isLoadingMore, sort, onSortChange }: {
  posts: Post[];
  systemTime: number;
  onLoadMore: () => void;
  hasMore: boolean;
  isLoadingMore: boolean;
  sort: SortKey;
  onSortChange: (sort: SortKey) => void;
}) {
  return (
    <div className="max-w-2xl mx-auto pt-24 pb-20 px-4">
      {/* 排序筛选栏 */}
      <div className="flex gap-2 overflow-x-auto pb-3 mb-3 -mx-4 px-4 sm:mx-0 sm:px-0 sticky top-14 z-10 bg-slate-50/85 backdrop-blur">
        {SORT_OPTIONS.map((opt) => (
          <button
            key={opt.value}
            onClick={() => onSortChange(opt.value)}
            className={clsx(
              "px-3.5 py-1.5 rounded-full text-sm font-medium whitespace-nowrap transition-colors border shrink-0",
              sort === opt.value
                ? "bg-cyan-600 text-white border-cyan-600 shadow-sm"
                : "bg-white text-slate-600 border-slate-200 hover:border-cyan-200 hover:text-cyan-600"
            )}
          >
            {opt.label}
          </button>
        ))}
      </div>

      <PostList posts={posts} systemTime={systemTime} onLoadMore={onLoadMore} hasMore={hasMore} isLoadingMore={isLoadingMore} />
    </div>
  );
}

function AppContent() {
  const [posts, setPosts] = useState<Post[]>([]);
  const [traces, setTraces] = useState<Trace[]>([]);
  const [, setErrorCount] = useState(0);
  const [pollInterval, setPollInterval] = useState(3000); // Changed to 3s
  const [systemTime, setSystemTime] = useState<number>(Date.now());
  const [systemTimeLabel, setSystemTimeLabel] = useState<string>('');
  const [timeMode, setTimeMode] = useState<string>('');
  const [hasMore, setHasMore] = useState(true);
  const [isLoadingMore, setIsLoadingMore] = useState(false);
  const [sort, setSort] = useState<SortKey>('new');

  const isPollingRef = useRef(false);
  // 已加载帖子数（用于排序模式下轮询时按当前长度重新拉取整页，避免「加载更多」被 3 秒轮询冲掉）
  const postsCountRef = useRef(0);
  useEffect(() => {
    postsCountRef.current = posts.length;
  }, [posts]);

  const handleSortChange = (next: SortKey) => {
    if (next === sort) return;
    setSort(next);
    setPosts([]);
    setHasMore(true);
    setIsLoadingMore(false);
  };

  const handleLoadMore = async () => {
    if (isLoadingMore || !hasMore) return;
    setIsLoadingMore(true);
    try {
      const morePosts = await getPosts(20, postsCountRef.current, sort);
      if (morePosts.length < 20) {
        setHasMore(false);
      }
      setPosts(prev => [...prev, ...morePosts]);
      postsCountRef.current += morePosts.length;
    } catch (error) {
      console.error("Failed to load more posts", error);
    } finally {
      setIsLoadingMore(false);
    }
  };

  useEffect(() => {
    const fetchData = async () => {
      if (isPollingRef.current) return;
      isPollingRef.current = true;

      try {
        // 排序模式下按当前已加载条数拉取整页（顺序随互动数变化，直接整体替换）；
        // 「最新」模式保持原有的「拉最新 20 条 + 头部合并」逻辑。
        const fetchLimit = sort === 'new' ? 20 : Math.max(20, postsCountRef.current);
        const [newPosts, newTraces, currentTimeData] = await Promise.all([
          getPosts(fetchLimit, 0, sort),
          getTraces(),
          getTime()
        ]);

        // Use system time for relative time calculation
        const systemTime = new Date(currentTimeData.current_time).getTime();

        if (sort === 'new') {
          setPosts(prev => {
            if (newPosts.length === 0) return prev;

            const prevIds = new Set(prev.map(p => p.id));
            const trulyNewPosts = newPosts.filter(p => !prevIds.has(p.id));
            const newPostsMap = new Map(newPosts.map(p => [p.id, p]));
            const updatedPrev = prev.map(p => newPostsMap.has(p.id) ? newPostsMap.get(p.id)! : p);
            return [...trulyNewPosts, ...updatedPrev];
          });
        } else {
          setPosts(newPosts);
        }

        setSystemTime(systemTime);
        setSystemTimeLabel(currentTimeData.current_time);
        setTimeMode(currentTimeData.mode);

        if (Array.isArray(newTraces)) {
          setTraces(prev => {
            if (newTraces.length === 0) return prev;

            const existingIds = new Set(prev.map(t => t.id));
            const uniqueNew = newTraces.filter(t => !existingIds.has(t.id));

            if (uniqueNew.length === 0) return prev;

            const updated = [...uniqueNew, ...prev];
            return updated.slice(0, 50);
          });
        }

        // Success - reset error count
        setErrorCount(0);
        if (pollInterval !== 3000) setPollInterval(3000);

      } catch (error) {
        console.error("Polling error", error);
        setErrorCount(prev => {
          const newCount = prev + 1;
          if (newCount >= 3) setPollInterval(15000);
          return newCount;
        });
      } finally {
        isPollingRef.current = false;
      }
    };

    fetchData(); // Fetch immediately on mount
    const id = setInterval(fetchData, pollInterval);
    return () => clearInterval(id);
  }, [pollInterval, sort]);

  return (
    <div className="min-h-screen bg-slate-50 text-slate-800 font-body">
      <SystemClock
        time={systemTimeLabel}
        mode={timeMode}
        className="left-auto right-4 md:right-6"
      />
      <TracePanel
        traces={traces}
      />

      <Routes>
        <Route path="/" element={<Home posts={posts} systemTime={systemTime} onLoadMore={handleLoadMore} hasMore={hasMore} isLoadingMore={isLoadingMore} sort={sort} onSortChange={handleSortChange} />} />
        <Route path="/post/:id" element={<PostDetail systemTime={systemTime} />} />
      </Routes>
    </div>
  );
}

function App() {
  return (
    <BrowserRouter>
      <AppContent />
    </BrowserRouter>
  );
}

export default App;
