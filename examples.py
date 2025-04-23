"""
Ví dụ về cách sử dụng Crawl4AI với từng loại trích xuất khác nhau
"""

import asyncio
import json
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode, LLMConfig
from crawl4ai.extraction_strategy import JsonCssExtractionStrategy, LLMExtractionStrategy
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy

async def example_markdown_crawl():
    """Ví dụ về cách trích xuất markdown đơn giản"""
    print("\n----- Ví dụ trích xuất Markdown -----")

    # Tạo bộ lọc và trình tạo markdown
    content_filter = PruningContentFilter(threshold=0.4, threshold_type="fixed")
    md_generator = DefaultMarkdownGenerator(content_filter=content_filter)
    
    # Tạo cấu hình
    browser_config = BrowserConfig(headless=True)
    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        markdown_generator=md_generator
    )
    
    # Thực hiện crawl
    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await crawler.arun(
            url="https://example.com",
            config=run_config
        )
        
        print(f"Raw markdown length: {len(result.markdown.raw_markdown)}")
        print(f"Fit markdown length: {len(result.markdown.fit_markdown)}")
        print(result.markdown.fit_markdown[:300] + "...")

async def example_css_extraction():
    """Ví dụ về cách trích xuất dữ liệu có cấu trúc sử dụng CSS selectors"""
    print("\n----- Ví dụ trích xuất CSS -----")
    
    # Định nghĩa schema CSS
    schema = {
        "name": "Example Items",
        "baseSelector": "body",
        "fields": [
            {"name": "title", "selector": "h1", "type": "text"},
            {"name": "description", "selector": "p", "type": "text"}
        ]
    }
    
    # Tạo cấu hình
    browser_config = BrowserConfig(headless=True)
    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        extraction_strategy=JsonCssExtractionStrategy(schema)
    )
    
    # Thực hiện crawl
    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await crawler.arun(
            url="https://example.com",
            config=run_config
        )
        
        # In kết quả JSON
        if result.extracted_content:
            data = json.loads(result.extracted_content)
            print(json.dumps(data, indent=2, ensure_ascii=False))
        else:
            print("Không có dữ liệu được trích xuất.")

async def example_dynamic_page_interaction():
    """Ví dụ về tương tác với trang động"""
    print("\n----- Ví dụ tương tác với trang động -----")
    
    # JavaScript để tự động cuộn trang và click các tab
    js_scroll_and_click = """
    (async () => {
        // Cuộn trang xuống cuối
        window.scrollTo(0, document.body.scrollHeight);
        await new Promise(r => setTimeout(r, 1000));
        
        // Nếu có các nút "Xem thêm", thử click chúng
        const loadMoreButtons = document.querySelectorAll('button:contains("Xem thêm"), a:contains("Xem thêm")');
        if (loadMoreButtons.length > 0) {
            for (let btn of loadMoreButtons) {
                btn.click();
                await new Promise(r => setTimeout(r, 1000));
            }
        }
        
        return "Hoàn thành tương tác";
    })();
    """
    
    # Tạo cấu hình
    browser_config = BrowserConfig(headless=False)  # headless=False để xem trình duyệt hoạt động
    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        js_code=[js_scroll_and_click],
        page_timeout=30000  # Tăng timeout lên 30 giây
    )
    
    # Thực hiện crawl
    async with AsyncWebCrawler(config=browser_config) as crawler:
        try:
            result = await crawler.arun(
                url="https://news.ycombinator.com/",  # Thử với Hacker News
                config=run_config
            )
            
            print(f"Độ dài markdown: {len(result.markdown)}")
            print(result.markdown[:300] + "...")
        except Exception as e:
            print(f"Lỗi: {str(e)}")

async def deep_crawl_bfs_example():
    print("\n=== Ví dụ Deep Crawl với BFS Strategy ===")
    
    # Tạo BFS Deep Crawl Strategy với giới hạn số trang
    bfs_strategy = BFSDeepCrawlStrategy(
        max_depth=2,         # Cào tối đa 2 cấp độ sâu
        max_pages=10,         # Giới hạn 10 trang
        include_external=False # Chỉ cào trong cùng domain
    )
    
    # Tạo CrawlerRunConfig
    config = CrawlerRunConfig(
        deep_crawl_strategy=bfs_strategy,
        stream=True,
        verbose=True
    )
    
    # Thực hiện deep crawl
    async with AsyncWebCrawler() as crawler:
        results = []
        count = 0
        
        print("Bắt đầu crawl sâu...")
        # Sử dụng streaming mode để xử lý từng kết quả khi nhận được
        async for result in await crawler.arun("https://example.com", config=config):
            count += 1
            depth = result.metadata.get('depth', 0) if hasattr(result, 'metadata') else 0
            
            # Lưu kết quả
            results.append({
                "url": result.url,
                "depth": depth,
                "title": result.title
            })
            
            # In thông tin
            print(f"URL {count}: {result.url} | Độ sâu: {depth}")
        
        print(f"\nĐã crawl tổng cộng {len(results)} trang.")
        print("Chi tiết về các trang đã crawl:")
        depths = {}
        for r in results:
            depths[r["depth"]] = depths.get(r["depth"], 0) + 1
        
        for depth, count in sorted(depths.items()):
            print(f"- Độ sâu {depth}: {count} trang")

async def main():
    """Hàm main chạy tất cả các ví dụ"""
    await example_markdown_crawl()
    await example_css_extraction()
    await example_dynamic_page_interaction()
    await deep_crawl_bfs_example()

if __name__ == "__main__":
    asyncio.run(main()) 