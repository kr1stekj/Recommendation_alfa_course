"""Команды для рекомендаций и интерактивное демо без браузерного интерфейса."""
import argparse
import json
import secrets
import uuid

from .engine import Recommender
from .database import Database
from .ranking import enable_ranking_logging


def main():
    """Разбирает аргументы и передаёт работу общему движку, который использует и API."""
    p = argparse.ArgumentParser(description="Альфа-Курс: локальный прототип рекомендаций")
    p.add_argument("command", choices=["recommend", "event", "demo", "cleanup", "merge"])
    p.add_argument("--artifacts", default="artifacts")
    p.add_argument("--task", default="")
    p.add_argument("--profile")
    p.add_argument("--item")
    p.add_argument("--event-type", default="click")
    p.add_argument("--event-id")
    p.add_argument("--target-profile")
    p.add_argument("--stage", default="")
    p.add_argument("--industry", default="")
    p.add_argument("--experience", default="")
    p.add_argument("--format", default="")
    p.add_argument("--topic", action="append")
    p.add_argument("--algorithm", choices=["hybrid", "topic", "cosine"], default="hybrid")
    p.add_argument("--llm", action="store_true")
    p.add_argument("--ranking-log", action="store_true", help="Числовые вклады ранжирования в stderr, без пользовательских текстов")
    p.add_argument("-k", type=int, default=5)
    args = p.parse_args()
    if args.ranking_log:
        enable_ranking_logging()
    # Обслуживание БД не зависит от каталога, модели или импорта history.
    if args.command == 'cleanup':
        database = Database()
        database.initialize()
        print(json.dumps({'expired_profiles_deleted': database.cleanup_expired()}, ensure_ascii=False, indent=2))
        return
    engine = Recommender(args.artifacts)
    if args.command == "event":
        result = engine.event(args.profile, args.item, args.event_type, args.event_id or str(uuid.uuid4()))
    elif args.command == "merge":
        engine.validate_profile_id(args.profile)
        engine.validate_profile_id(args.target_profile)
        result = engine.store.merge(args.profile, args.target_profile)
    elif args.command == "demo":
        # Случайный идентификатор различает гостей, но не раскрывает личность человека.
        profile = args.profile or "anon:" + secrets.token_urlsafe(16)
        task = args.task
        print("Демо-профиль:", profile, "| Эмбеддинги:", engine.encoder.config["backend"])
        print("Это локальная симуляция кликов. Пустой запрос показывает стартовую подборку.")
        try:
            while True:
                result = engine.recommend(task=task, profile_id=profile, k=args.k, use_llm=args.llm)
                print("\nИнтересы:", result["profile_topics"], "| Режим:", result["mode"])
                if args.llm:
                    print("Проверка рекомендаций:", result['llm_status'])
                for i, item in enumerate(result["items"], 1):
                    price = "бесплатно" if item["price"] == 0 else str(item["price"]) + " руб." if item["price"] else "цена/доступ не определены"
                    print(f'{i}. {item["title"]} [{item["kind"]}, {price}]\n   {item["explanation"]}\n   {item["url"]}')
                action = input("\nНомер = клик; ?текст = задача; q = выход: ").strip()
                if action == "q": break
                if action.startswith("?"): task = action[1:].strip()
                elif action.isdigit() and 1 <= int(action) <= len(result["items"]):
                    # Номер карточки симулирует вовлечённый просмотр; это не реальный веб-лог.
                    engine.event(profile, result["items"][int(action)-1]["id"], "engaged", str(uuid.uuid4()))
        except (EOFError, KeyboardInterrupt):
            pass
        return
    else:
        result = engine.recommend(task=args.task, profile_id=args.profile, stage=args.stage, industry=args.industry,
                                  experience=args.experience, preferred_format=args.format, topics=args.topic,
                                  k=args.k, algorithm=args.algorithm, use_llm=args.llm)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
